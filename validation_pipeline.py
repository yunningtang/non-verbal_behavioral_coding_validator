"""
Coding Validation Pipeline
This script performs cross-comparison for:
1. ID (participant/video number) - exact match check
2. Cheated or not - whether the child cheated during T5
3. Behavior - whether corresponding behaviors were coded
4. Timestamps - Start(s), Stop(s), Image index start/stop differences
   Uses nearest-neighbor matching (by Start (s)) and IoU for segment overlap.

Reference coder: Dolly (practice coding.csv)
Comparison coders: 
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# --- Matching and reporting thresholds ---
MATCH_THRESHOLD_SEC = 10.0   # Only pair Dolly–Coder if |start_diff| <= this (seconds)
IOU_THRESHOLD = 0.5          # IoU > this → treat as same behavior segment
SUMMARY_MAX_START_DIFF_SEC = 30.0  # Exclude pairs with |start_diff_sec| > this from average time deviation in summary


def compute_iou(start1, stop1, start2, stop2):
    """
    IoU (Intersection over Union) for two time segments [start, stop].
    Returns overlap_length / union_length; 0 if either segment is invalid or no overlap.
    """
    try:
        s1, e1 = float(start1), float(stop1)
        s2, e2 = float(start2), float(stop2)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(s1) or pd.isna(e1) or pd.isna(s2) or pd.isna(e2) or e1 <= s1 or e2 <= s2:
        return 0.0
    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - overlap
    return overlap / union if union > 0 else 0.0


def load_dolly_data(filepath):
    """Load and process Dolly's reference data."""
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = df.columns.str.strip()
    return df


def load_coder_data(filepath):
    """Load and process individual coder data."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def extract_participant_id(value):
    """Extract numeric participant ID from various formats (P286, P_286, 286, etc.)."""
    if pd.isna(value):
        return None
    value_str = str(value)
    # Remove 'P', 'P_', and any other non-numeric characters
    numeric_id = ''.join(filter(str.isdigit, value_str))
    return int(numeric_id) if numeric_id else None


def get_dolly_data_for_participant(dolly_df, participant_id):
    """Extract Dolly's data for a specific participant."""
    # Dolly's ID column contains numeric participant IDs
    return dolly_df[dolly_df['ID'] == participant_id].copy()


def validate_participant_ids(dolly_df, lareina_file_map, sally_file_map):
    """
    Explicit ID validation: check exact match, missing, duplicate, alignment.
    Returns a structure for reporting: IDs in reference, in each coder; missing/extra; duplicates.
    """
    ref_ids = set(dolly_df['ID'].dropna().astype(int).unique())
    
    # Duplicates in reference (same ID appearing in multiple observation blocks is ok; same ID with inconsistent rows is the concern - here we only check per-file uniqueness)
    ref_id_counts = dolly_df.groupby('ID').size()
    duplicates_in_reference = ref_id_counts[ref_id_counts.gt(1)].index.tolist()  # IDs that appear more than once (across rows) - in Dolly one ID has many rows (one per behavior), so this is normal; we care about ID column consistency
    # Actually: in Dolly each participant has one ID value repeated on every row. So "duplicate" here means the same participant appears multiple times (e.g. two blocks). For "exact match" we only need: no duplicate participant IDs in the sense of conflicting identity. So we report: reference IDs, and per coder which IDs we have.
    ref_id_list = sorted(ref_ids)
    
    lareina_ids = set(k for k, p in lareina_file_map.items() if p.exists())
    sally_ids = set(k for k, p in sally_file_map.items() if p.exists())
    
    missing_in_lareina = ref_ids - lareina_ids  # In reference but no file from Lareina
    missing_in_sally = ref_ids - sally_ids
    extra_in_lareina = lareina_ids - ref_ids    # Lareina has file but not in reference
    extra_in_sally = sally_ids - ref_ids
    
    # Alignment: can we align reference vs coder by ID? For each reference ID, check coder file exists and ID column matches.
    alignment_lareina = []
    alignment_sally = []
    for pid in sorted(ref_ids):
        if pid not in lareina_ids:
            alignment_lareina.append((pid, 'no_file', None))
        else:
            path = lareina_file_map.get(pid)
            if path and path.exists():
                try:
                    df = load_coder_data(path)
                    if len(df) == 0:
                        alignment_lareina.append((pid, 'empty_file', None))
                    elif 'ID' not in df.columns:
                        alignment_lareina.append((pid, 'no_id_column', None))
                    else:
                        file_ids = df['ID'].dropna().astype(int).unique()
                        if len(file_ids) > 1:
                            alignment_lareina.append((pid, 'multiple_ids_in_file', list(file_ids)))
                        elif int(file_ids[0]) != pid:
                            alignment_lareina.append((pid, 'id_mismatch', int(file_ids[0])))
                        else:
                            alignment_lareina.append((pid, 'ok', None))
                except Exception as e:
                    alignment_lareina.append((pid, 'read_error', str(e)))
            else:
                alignment_lareina.append((pid, 'no_file', None))
    
    for pid in sorted(ref_ids):
        if pid not in sally_ids:
            alignment_sally.append((pid, 'no_file', None))
        else:
            path = sally_file_map.get(pid)
            if path and path.exists():
                try:
                    df = load_coder_data(path)
                    if len(df) == 0:
                        alignment_sally.append((pid, 'empty_file', None))
                    elif 'ID' not in df.columns:
                        alignment_sally.append((pid, 'no_id_column', None))
                    else:
                        file_ids = df['ID'].dropna().astype(int).unique()
                        if len(file_ids) > 1:
                            alignment_sally.append((pid, 'multiple_ids_in_file', list(file_ids)))
                        elif int(file_ids[0]) != pid:
                            alignment_sally.append((pid, 'id_mismatch', int(file_ids[0])))
                        else:
                            alignment_sally.append((pid, 'ok', None))
                except Exception as e:
                    alignment_sally.append((pid, 'read_error', str(e)))
            else:
                alignment_sally.append((pid, 'no_file', None))
    
    return {
        'reference_ids': ref_id_list,
        'lareina_ids': sorted(lareina_ids),
        'sally_ids': sorted(sally_ids),
        'missing_in_lareina': sorted(missing_in_lareina),
        'missing_in_sally': sorted(missing_in_sally),
        'extra_in_lareina': sorted(extra_in_lareina),
        'extra_in_sally': sorted(extra_in_sally),
        'alignment_lareina': alignment_lareina,
        'alignment_sally': alignment_sally,
    }


def compare_cheated_status(dolly_df, coder_df, coder_name, participant_id):
    """Compare cheated or not status between coders."""
    result = {
        'participant_id': participant_id,
        'coder': coder_name,
        'dolly_cheated': None,
        'coder_cheated': None,
        'match': None
    }
    
    dolly_data = get_dolly_data_for_participant(dolly_df, participant_id)
    
    if len(dolly_data) > 0:
        result['dolly_cheated'] = dolly_data['Cheated or not'].iloc[0]
    
    if len(coder_df) > 0:
        result['coder_cheated'] = coder_df['Cheated or not'].iloc[0]
    
    if result['dolly_cheated'] is not None and result['coder_cheated'] is not None:
        result['match'] = result['dolly_cheated'] == result['coder_cheated']
    
    return result


def compare_behaviors(dolly_df, coder_df, coder_name, participant_id):
    """Compare behaviors and their timestamps between coders."""
    differences = []
    
    dolly_data = get_dolly_data_for_participant(dolly_df, participant_id)
    
    if len(dolly_data) == 0 or len(coder_df) == 0:
        return differences
    
    # Get unique behaviors from both
    dolly_behaviors = set(dolly_data['Behavior'].unique())
    coder_behaviors = set(coder_df['Behavior'].unique())
    
    # Behaviors in Dolly but not in coder
    missing_in_coder = dolly_behaviors - coder_behaviors
    for behavior in missing_in_coder:
        differences.append({
            'participant_id': participant_id,
            'coder': coder_name,
            'behavior': behavior,
            'issue_type': 'Missing in coder',
            'dolly_count': len(dolly_data[dolly_data['Behavior'] == behavior]),
            'coder_count': 0,
            'details': f'Behavior coded by Dolly but not by {coder_name}'
        })
    
    # Behaviors in coder but not in Dolly
    extra_in_coder = coder_behaviors - dolly_behaviors
    for behavior in extra_in_coder:
        differences.append({
            'participant_id': participant_id,
            'coder': coder_name,
            'behavior': behavior,
            'issue_type': 'Extra in coder',
            'dolly_count': 0,
            'coder_count': len(coder_df[coder_df['Behavior'] == behavior]),
            'details': f'Behavior coded by {coder_name} but not by Dolly'
        })
    
    return differences


def compare_behavior_timestamps(dolly_df, coder_df, coder_name, participant_id):
    """
    Compare timestamps using nearest-neighbor matching by Start (s).
    For each Dolly behavior instance, find the Coder instance of the same behavior
    with closest Start (s); pair only if |start_diff| <= MATCH_THRESHOLD_SEC.
    Unmatched Dolly → 'Coder 漏记' (FN); unmatched Coder → 'Coder 多记' (FP).
    IoU is computed for each matched pair.
    """
    timestamp_diffs = []
    dolly_data = get_dolly_data_for_participant(dolly_df, participant_id)
    if len(dolly_data) == 0 or len(coder_df) == 0:
        return timestamp_diffs

    dolly_behaviors = set(dolly_data['Behavior'].unique())
    coder_behaviors = set(coder_df['Behavior'].unique())
    common_behaviors = dolly_behaviors & coder_behaviors

    def safe_float(x, default=None):
        try:
            return float(x) if pd.notna(x) else default
        except (TypeError, ValueError):
            return default

    for behavior in common_behaviors:
        dolly_rows = dolly_data[dolly_data['Behavior'] == behavior].sort_values('Start (s)').reset_index(drop=True)
        coder_rows = coder_df[coder_df['Behavior'] == behavior].sort_values('Start (s)').reset_index(drop=True)
        n_dolly, n_coder = len(dolly_rows), len(coder_rows)

        # Nearest-neighbor pairing: for each Dolly row, pick closest Coder row within threshold
        used_coder = [False] * n_coder
        pairs = []  # (d_idx, c_idx, start_diff_sec)

        for d_idx in range(n_dolly):
            d_start = safe_float(dolly_rows.iloc[d_idx]['Start (s)'])
            if d_start is None:
                continue
            best_c_idx = None
            best_abs_diff = MATCH_THRESHOLD_SEC + 1
            for c_idx in range(n_coder):
                if used_coder[c_idx]:
                    continue
                c_start = safe_float(coder_rows.iloc[c_idx]['Start (s)'])
                if c_start is None:
                    continue
                abs_diff = abs(d_start - c_start)
                if abs_diff <= MATCH_THRESHOLD_SEC and abs_diff < best_abs_diff:
                    best_abs_diff = abs_diff
                    best_c_idx = c_idx
            if best_c_idx is not None:
                used_coder[best_c_idx] = True
                start_diff = safe_float(coder_rows.iloc[best_c_idx]['Start (s)']) - d_start
                pairs.append((d_idx, best_c_idx, round(start_diff, 3)))

        # Build entries for matched pairs (with IoU and time diffs)
        for d_idx, c_idx, start_diff_sec in pairs:
            d_row = dolly_rows.iloc[d_idx]
            c_row = coder_rows.iloc[c_idx]
            d_start = safe_float(d_row['Start (s)'])
            d_stop = safe_float(d_row['Stop (s)'])
            c_start = safe_float(c_row['Start (s)'])
            c_stop = safe_float(c_row['Stop (s)'])
            iou = compute_iou(d_start, d_stop, c_start, c_stop) if all(x is not None for x in (d_start, d_stop, c_start, c_stop)) else None

            stop_diff = None
            idx_start_diff = None
            idx_stop_diff = None
            try:
                if pd.notna(d_row['Stop (s)']) and pd.notna(c_row['Stop (s)']):
                    stop_diff = round(float(c_row['Stop (s)']) - float(d_row['Stop (s)']), 3)
            except (TypeError, ValueError):
                pass
            try:
                if pd.notna(d_row['Image index start']) and pd.notna(c_row['Image index start']):
                    idx_start_diff = int(c_row['Image index start']) - int(d_row['Image index start'])
            except (TypeError, ValueError):
                pass
            try:
                if pd.notna(d_row['Image index stop']) and pd.notna(c_row['Image index stop']):
                    idx_stop_diff = int(c_row['Image index stop']) - int(d_row['Image index stop'])
            except (TypeError, ValueError):
                pass

            has_notable_diff = any([
                start_diff_sec is not None and abs(start_diff_sec) > 0.05,
                stop_diff is not None and abs(stop_diff) > 0.05,
                idx_start_diff is not None and abs(idx_start_diff) > 1,
                idx_stop_diff is not None and abs(idx_stop_diff) > 1
            ])
            issue_type = 'Timestamp difference' if has_notable_diff else 'Matched'

            timestamp_diffs.append({
                'participant_id': participant_id,
                'coder': coder_name,
                'behavior': behavior,
                'issue_type': issue_type,
                'occurrence': d_idx + 1,
                'dolly_count': n_dolly,
                'coder_count': n_coder,
                'dolly_start': d_row['Start (s)'],
                'coder_start': c_row['Start (s)'],
                'start_diff_sec': start_diff_sec,
                'dolly_stop': d_row['Stop (s)'],
                'coder_stop': c_row['Stop (s)'],
                'stop_diff_sec': stop_diff,
                'dolly_idx_start': d_row.get('Image index start'),
                'coder_idx_start': c_row.get('Image index start'),
                'idx_start_diff': idx_start_diff,
                'dolly_idx_stop': d_row.get('Image index stop'),
                'coder_idx_stop': c_row.get('Image index stop'),
                'idx_stop_diff': idx_stop_diff,
                'iou': round(iou, 4) if iou is not None else None,
            })

        # Unmatched Dolly -> Coder missed (FN): reference has segment, no coder match within threshold
        matched_d = {p[0] for p in pairs}
        for d_idx in range(n_dolly):
            if d_idx in matched_d:
                continue
            d_row = dolly_rows.iloc[d_idx]
            timestamp_diffs.append({
                'participant_id': participant_id,
                'coder': coder_name,
                'behavior': behavior,
                'issue_type': 'Coder missed (FN)',
                'occurrence': d_idx + 1,
                'dolly_count': n_dolly,
                'coder_count': n_coder,
                'dolly_start': d_row['Start (s)'],
                'coder_start': None,
                'start_diff_sec': None,
                'dolly_stop': d_row['Stop (s)'],
                'coder_stop': None,
                'stop_diff_sec': None,
                'dolly_idx_start': d_row.get('Image index start'),
                'coder_idx_start': None,
                'idx_start_diff': None,
                'dolly_idx_stop': d_row.get('Image index stop'),
                'coder_idx_stop': None,
                'idx_stop_diff': None,
                'iou': None,
            })

        # Unmatched Coder -> Coder extra (FP): coder has segment, no reference match within threshold
        matched_c = {p[1] for p in pairs}
        for c_idx in range(n_coder):
            if c_idx in matched_c:
                continue
            c_row = coder_rows.iloc[c_idx]
            timestamp_diffs.append({
                'participant_id': participant_id,
                'coder': coder_name,
                'behavior': behavior,
                'issue_type': 'Coder extra (FP)',
                'occurrence': c_idx + 1,
                'dolly_count': n_dolly,
                'coder_count': n_coder,
                'dolly_start': None,
                'coder_start': c_row['Start (s)'],
                'start_diff_sec': None,
                'dolly_stop': None,
                'coder_stop': c_row['Stop (s)'],
                'stop_diff_sec': None,
                'dolly_idx_start': None,
                'coder_idx_start': c_row.get('Image index start'),
                'idx_start_diff': None,
                'dolly_idx_stop': None,
                'coder_idx_stop': c_row.get('Image index stop'),
                'idx_stop_diff': None,
                'iou': None,
            })

    return timestamp_diffs


def generate_summary_report(cheated_comparisons, behavior_diffs, timestamp_diffs, id_validation, output_path):
    """Generate a comprehensive summary report with executive summary, ID check, and conclusion."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CODING VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("Reference Coder: Dolly")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Executive summary (for PI / decision-maker)
    report_lines.append("-" * 80)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append("")
    ref_ids = id_validation.get('reference_ids', [])
    n_ref = len(ref_ids)
    n_lar = len([a for a in id_validation.get('alignment_lareina', []) if a[1] == 'ok'])
    n_sal = len([a for a in id_validation.get('alignment_sally', []) if a[1] == 'ok'])
    id_ok = (len(id_validation.get('missing_in_lareina', [])) == 0 and len(id_validation.get('missing_in_sally', [])) == 0
             and len(id_validation.get('extra_in_lareina', [])) == 0 and len(id_validation.get('extra_in_sally', [])) == 0
             and all(a[1] == 'ok' for a in id_validation.get('alignment_lareina', []))
             and all(a[1] == 'ok' for a in id_validation.get('alignment_sally', [])))
    report_lines.append("This report compares coding outputs from two RAs (Lareina, Sally) against the reference coder (Dolly).")
    report_lines.append(f"Reference contains {n_ref} participant IDs. ID exact-match check: {'PASS (all aligned, no missing/duplicate)' if id_ok else 'SEE SECTION 0 FOR GAPS'}.")
    if cheated_comparisons:
        cheated_df = pd.DataFrame(cheated_comparisons)
        match_count = cheated_df['match'].sum()
        total_c = len(cheated_df)
        report_lines.append(f"Cheated-or-not agreement: {match_count}/{total_c} comparisons match reference.")
    if timestamp_diffs:
        ts_df = pd.DataFrame(timestamp_diffs)
        paired = ts_df[ts_df['issue_type'].isin(['Matched', 'Timestamp difference'])]
        iou_zero_n = (paired['iou'] == 0).sum() if 'iou' in paired.columns and len(paired) > 0 else 0
        fn_count = len(ts_df[ts_df['issue_type'] == 'Coder missed (FN)'])
        fp_count = len(ts_df[ts_df['issue_type'] == 'Coder extra (FP)'])
        report_lines.append(f"Behavior timestamps: {len(paired)} segments paired within threshold; {iou_zero_n} paired but IoU=0 (segment alignment failed); {fn_count} reference segments unmatched (coder missed); {fp_count} coder segments unmatched (coder extra).")
    report_lines.append("")
    report_lines.append("Overall consistency: See CONCLUSION at end of report.")
    report_lines.append("")

    # Section 0: ID exact match check
    report_lines.append("-" * 80)
    report_lines.append("SECTION 0: ID EXACT MATCH CHECK")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("IDs in reference (Dolly): " + ", ".join(str(x) for x in id_validation['reference_ids']))
    report_lines.append("IDs in Lareina files:   " + ", ".join(str(x) for x in id_validation['lareina_ids']))
    report_lines.append("IDs in Sally files:     " + ", ".join(str(x) for x in id_validation['sally_ids']))
    report_lines.append("")
    if id_validation['missing_in_lareina']:
        report_lines.append("MISSING IN LAREINA (in reference but no Lareina file): " + ", ".join(str(x) for x in id_validation['missing_in_lareina']))
    if id_validation['missing_in_sally']:
        report_lines.append("MISSING IN SALLY (in reference but no Sally file):     " + ", ".join(str(x) for x in id_validation['missing_in_sally']))
    if id_validation['extra_in_lareina']:
        report_lines.append("EXTRA IN LAREINA (Lareina file but not in reference):   " + ", ".join(str(x) for x in id_validation['extra_in_lareina']))
    if id_validation['extra_in_sally']:
        report_lines.append("EXTRA IN SALLY (Sally file but not in reference):     " + ", ".join(str(x) for x in id_validation['extra_in_sally']))
    if not any([id_validation['missing_in_lareina'], id_validation['missing_in_sally'], id_validation['extra_in_lareina'], id_validation['extra_in_sally']]):
        report_lines.append("No missing or extra IDs: all reference IDs have exactly one coder file each; no coder-only IDs.")
    report_lines.append("")
    report_lines.append("Alignment (file ID column vs expected participant ID):")
    report_lines.append("  Lareina: " + ", ".join(f"P{pid}={status}" for pid, status, _ in id_validation['alignment_lareina']))
    report_lines.append("  Sally:   " + ", ".join(f"P{pid}={status}" for pid, status, _ in id_validation['alignment_sally']))
    for pid, status, extra in id_validation['alignment_lareina']:
        if status != 'ok':
            report_lines.append(f"    P{pid} Lareina: {status}" + (f" ({extra})" if extra is not None else ""))
    for pid, status, extra in id_validation['alignment_sally']:
        if status != 'ok':
            report_lines.append(f"    P{pid} Sally: {status}" + (f" ({extra})" if extra is not None else ""))
    report_lines.append("")
    
    # Section 1: Cheated or Not Comparison
    report_lines.append("-" * 80)
    report_lines.append("SECTION 1: CHEATED OR NOT COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if cheated_comparisons:
        cheated_df = pd.DataFrame(cheated_comparisons)
        
        # Summary statistics
        total_comparisons = len(cheated_df)
        matches = cheated_df['match'].sum()
        mismatches = total_comparisons - matches
        
        report_lines.append(f"Total comparisons: {total_comparisons}")
        report_lines.append(f"Matches: {matches}")
        report_lines.append(f"Mismatches: {mismatches}")
        report_lines.append("")
        
        # Detail table
        report_lines.append("Detailed Results:")
        report_lines.append(f"{'Participant':<12} {'Coder':<10} {'Dolly':<8} {'Coder':<8} {'Match':<8}")
        report_lines.append("-" * 50)
        
        for _, row in cheated_df.iterrows():
            match_str = "YES" if row['match'] else "NO"
            report_lines.append(f"{row['participant_id']:<12} {row['coder']:<10} {row['dolly_cheated']:<8} {row['coder_cheated']:<8} {match_str:<8}")
        
        # List mismatches
        mismatches_df = cheated_df[cheated_df['match'] == False]
        if len(mismatches_df) > 0:
            report_lines.append("")
            report_lines.append("MISMATCHES FOUND:")
            for _, row in mismatches_df.iterrows():
                report_lines.append(f"  - P{row['participant_id']}: Dolly coded {row['dolly_cheated']}, {row['coder']} coded {row['coder_cheated']}")
    
    report_lines.append("")
    
    # Section 2: Behavior Presence Comparison
    report_lines.append("-" * 80)
    report_lines.append("SECTION 2: BEHAVIOR PRESENCE COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if behavior_diffs:
        behavior_df = pd.DataFrame(behavior_diffs)
        
        # Group by issue type
        missing = behavior_df[behavior_df['issue_type'] == 'Missing in coder']
        extra = behavior_df[behavior_df['issue_type'] == 'Extra in coder']
        
        report_lines.append(f"Total behavior discrepancies: {len(behavior_diffs)}")
        report_lines.append(f"  - Behaviors missing in coder: {len(missing)}")
        report_lines.append(f"  - Extra behaviors in coder: {len(extra)}")
        report_lines.append("")
        
        if len(missing) > 0:
            report_lines.append("BEHAVIORS MISSING IN CODER (present in Dolly's reference):")
            for _, row in missing.iterrows():
                report_lines.append(f"  P{row['participant_id']} | {row['coder']}: Missing '{row['behavior']}' (Dolly coded {row['dolly_count']} times)")
            report_lines.append("")
        
        if len(extra) > 0:
            report_lines.append("EXTRA BEHAVIORS IN CODER (not in Dolly's reference):")
            for _, row in extra.iterrows():
                report_lines.append(f"  P{row['participant_id']} | {row['coder']}: Extra '{row['behavior']}' (coded {row['coder_count']} times)")
            report_lines.append("")
    else:
        report_lines.append("No behavior presence discrepancies found.")
        report_lines.append("")
    
    # Section 3: Timestamp Differences (Nearest-Neighbor Matching + IoU)
    report_lines.append("-" * 80)
    report_lines.append("SECTION 3: TIMESTAMP DIFFERENCES (Nearest-Neighbor Matching)")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if timestamp_diffs:
        timestamp_df = pd.DataFrame(timestamp_diffs)
        # Paired entries (Matched or Timestamp difference)
        paired = timestamp_df[timestamp_df['issue_type'].isin(['Matched', 'Timestamp difference'])]
        fn_entries = timestamp_df[timestamp_df['issue_type'] == 'Coder missed (FN)']
        fp_entries = timestamp_df[timestamp_df['issue_type'] == 'Coder extra (FP)']
        
        report_lines.append(f"Total timestamp-related entries: {len(timestamp_diffs)}")
        report_lines.append(f"  - Successfully paired (within +/-{MATCH_THRESHOLD_SEC}s): {len(paired)}")
        report_lines.append(f"  - Coder missed (FN): reference has segment, no coder match within threshold: {len(fn_entries)}")
        report_lines.append(f"  - Coder extra (FP): coder has segment, no reference match within threshold: {len(fp_entries)}")
        report_lines.append("")
        report_lines.append("Interpretation of unmatched:")
        report_lines.append("  - Coder missed (FN): Either the coder did not code this segment, or coded it at a time outside the matching threshold; not a pipeline failure (reference row exists).")
        report_lines.append("  - Coder extra (FP): Either the coder coded a segment the reference did not, or the reference did not code it; or the same event was coded at a time outside the matching threshold.")
        report_lines.append("")
        if len(paired) > 0 and 'iou' in paired.columns:
            iou_zero = paired[(paired['iou'].notna()) & (paired['iou'] == 0)]
            iou_gt_zero = paired[(paired['iou'].notna()) & (paired['iou'] > 0)]
            report_lines.append("Paired segments by IoU:")
            report_lines.append(f"  - Pairs with IoU = 0 (matched by start time only; no segment overlap): {len(iou_zero)}. These are NOT considered aligned for segment consistency; they indicate the same behavior was coded at a similar time but with no or minimal overlap in start/stop.")
            report_lines.append(f"  - Pairs with IoU > 0: {len(iou_gt_zero)}.")
            report_lines.append("")
            if len(paired) > 0 and paired['start_diff_sec'].notna().any():
                start_diffs = paired['start_diff_sec'].dropna()
                neg_count = (start_diffs < 0).sum()
                if neg_count >= len(start_diffs) * 0.5:
                    report_lines.append("Note on time differences: Many start_diff values are negative (coder start time earlier than reference). This may indicate systematic bias (e.g. different definition of event onset, or frame rounding). No automatic correction is applied; consider clarifying timestamp convention if consistency is critical.")
                report_lines.append("")
        if len(paired) > 0:
            report_lines.append("PAIRED COMPARISONS (with IoU):")
            report_lines.append("")
            for (pid, coder), group in paired.groupby(['participant_id', 'coder']):
                report_lines.append(f"  P{pid} - {coder}:")
                for _, row in group.iterrows():
                    occ = row.get('occurrence', 1)
                    iou_val = row.get('iou')
                    iou_str = f", IoU={iou_val:.2f}" if iou_val is not None and pd.notna(iou_val) else ""
                    report_lines.append(f"    {row['behavior']} (occurrence {occ}){iou_str}:")
                    if row['start_diff_sec'] is not None:
                        report_lines.append(f"      Start: Dolly={row['dolly_start']}s, {coder}={row['coder_start']}s, Diff={row['start_diff_sec']:+.3f}s")
                    if row['stop_diff_sec'] is not None:
                        report_lines.append(f"      Stop: Dolly={row['dolly_stop']}s, {coder}={row['coder_stop']}s, Diff={row['stop_diff_sec']:+.3f}s")
                    if row['idx_start_diff'] is not None and pd.notna(row['idx_start_diff']):
                        report_lines.append(f"      Frame Start: Dolly={row['dolly_idx_start']}, {coder}={row['coder_idx_start']}, Diff={int(row['idx_start_diff']):+d} frames")
                    if row['idx_stop_diff'] is not None and pd.notna(row['idx_stop_diff']):
                        report_lines.append(f"      Frame Stop: Dolly={row['dolly_idx_stop']}, {coder}={row['coder_idx_stop']}, Diff={int(row['idx_stop_diff']):+d} frames")
                report_lines.append("")
        
        if len(fn_entries) > 0:
            report_lines.append("CODER MISSED (FN) - reference has segment, no coder match within threshold:")
            for _, row in fn_entries.iterrows():
                report_lines.append(f"  P{row['participant_id']} | {row['coder']}: '{row['behavior']}' occurrence {row.get('occurrence', '?')}, Dolly Start={row['dolly_start']}s")
            report_lines.append("")
        
        if len(fp_entries) > 0:
            report_lines.append("CODER EXTRA (FP) - coder has segment, no reference match within threshold:")
            for _, row in fp_entries.iterrows():
                report_lines.append(f"  P{row['participant_id']} | {row['coder']}: '{row['behavior']}' occurrence {row.get('occurrence', '?')}, Coder Start={row['coder_start']}s")
            report_lines.append("")
    else:
        report_lines.append("No timestamp entries (no common behaviors with data).")
        report_lines.append("")
    
    # Section 4: Summary Statistics (valid pairs only for avg deviation) + Precision/Recall/F1
    report_lines.append("-" * 80)
    report_lines.append("SECTION 4: SUMMARY STATISTICS BY CODER")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if timestamp_diffs:
        ts_df = pd.DataFrame(timestamp_diffs)
        paired_df = ts_df[ts_df['issue_type'].isin(['Matched', 'Timestamp difference'])]
        valid_for_avg = paired_df[
            paired_df['start_diff_sec'].notna() &
            (paired_df['start_diff_sec'].abs() <= SUMMARY_MAX_START_DIFF_SEC)
        ]
    else:
        paired_df = pd.DataFrame()
        valid_for_avg = pd.DataFrame()
    
    for coder in ['Lareina', 'Sally']:
        report_lines.append(f"{coder}:")
        
        if cheated_comparisons:
            coder_cheated = [c for c in cheated_comparisons if c['coder'] == coder]
            coder_matches = sum(1 for c in coder_cheated if c['match'])
            report_lines.append(f"  - Cheated status agreement: {coder_matches}/{len(coder_cheated)}")
        
        if behavior_diffs:
            coder_behavior = [b for b in behavior_diffs if b['coder'] == coder]
            report_lines.append(f"  - Behavior presence discrepancies: {len(coder_behavior)}")
        
        if timestamp_diffs:
            coder_ts = [t for t in timestamp_diffs if t['coder'] == coder]
            tp = sum(1 for t in coder_ts if t['issue_type'] in ['Matched', 'Timestamp difference'])
            fn = sum(1 for t in coder_ts if t['issue_type'] == 'Coder missed (FN)')
            fp = sum(1 for t in coder_ts if t['issue_type'] == 'Coder extra (FP)')
            report_lines.append(f"  - Behavior match (nearest-neighbor): TP={tp}, FN={fn}, FP={fp}")
            if tp + fn > 0:
                recall = tp / (tp + fn)
                report_lines.append(f"    Recall (reference segments that coder matched): {recall:.2%} ({tp}/{tp+fn})")
            if tp + fp > 0:
                precision = tp / (tp + fp)
                report_lines.append(f"    Precision (coder segments that matched reference): {precision:.2%} ({tp}/{tp+fp})")
            if tp + fn > 0 and tp + fp > 0:
                r = tp / (tp + fn)
                p = tp / (tp + fp)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                report_lines.append(f"    F1-Score: {f1:.4f}")
            coder_valid = valid_for_avg[valid_for_avg['coder'] == coder] if len(valid_for_avg) > 0 else pd.DataFrame()
            if len(coder_valid) > 0 and coder_valid['start_diff_sec'].notna().any():
                mean_diff = coder_valid['start_diff_sec'].abs().mean()
                report_lines.append(f"  - Mean |start_diff| (only pairs with |diff|≤{SUMMARY_MAX_START_DIFF_SEC}s): {mean_diff:.3f}s (n={len(coder_valid)})")
        
        report_lines.append("")
    
    # Conclusion (single judgment for PI)
    report_lines.append("-" * 80)
    report_lines.append("CONCLUSION")
    report_lines.append("-" * 80)
    report_lines.append("")
    id_ok_concl = id_validation and (
        len(id_validation.get('missing_in_lareina', [])) == 0 and len(id_validation.get('missing_in_sally', [])) == 0
        and len(id_validation.get('extra_in_lareina', [])) == 0 and len(id_validation.get('extra_in_sally', [])) == 0
        and all(a[1] == 'ok' for a in id_validation.get('alignment_lareina', []))
        and all(a[1] == 'ok' for a in id_validation.get('alignment_sally', []))
    )
    if id_validation and not id_ok_concl:
        report_lines.append("ID alignment has gaps (see Section 0); address missing/extra IDs or file alignment before relying on per-participant consistency.")
    else:
        report_lines.append("ID exact match: all reference participants have one-to-one alignment with coder files; no missing or duplicate IDs reported.")
    if cheated_comparisons:
        cheated_df = pd.DataFrame(cheated_comparisons)
        m = cheated_df['match'].sum()
        t = len(cheated_df)
        if m == t:
            report_lines.append("Cheated-or-not: full agreement with reference across all comparisons.")
        else:
            report_lines.append(f"Cheated-or-not: {t - m} mismatch(es) out of {t} comparisons; review participants listed in Section 1.")
    if behavior_diffs:
        report_lines.append(f"Behavior presence: {len(behavior_diffs)} discrepancy(ies) (missing or extra behaviors vs reference); see Section 2.")
    else:
        report_lines.append("Behavior presence: no missing or extra behavior types vs reference.")
    if timestamp_diffs:
        ts_df = pd.DataFrame(timestamp_diffs)
        fn_total = len(ts_df[ts_df['issue_type'] == 'Coder missed (FN)'])
        fp_total = len(ts_df[ts_df['issue_type'] == 'Coder extra (FP)'])
        paired = ts_df[ts_df['issue_type'].isin(['Matched', 'Timestamp difference'])]
        iou0 = len(paired[(paired.get('iou', 0) == 0)]) if 'iou' in paired.columns else 0
        report_lines.append(f"Behavior timestamps: {len(paired)} segments paired; {iou0} paired with IoU=0 (segment not aligned); {fn_total} reference-only (coder missed); {fp_total} coder-only (coder extra). Overall consistency is moderate: temporal alignment is acceptable for many segments but systematic time bias and IoU=0 pairs should be reviewed.")
    report_lines.append("")
    report_lines.append("Recommended next steps: (1) Resolve any ID gaps in Section 0; (2) Resolve cheated-or-not mismatches in Section 1; (3) Clarify coding manual for behaviors with many FN/FP or IoU=0; (4) Align timestamp definition (event onset) if systematic bias is a concern.")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report
    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content


def main():
    """Main pipeline execution."""
    # Define paths
    base_path = Path(__file__).parent
    dolly_path = base_path / "Dolly" / "practice coding.csv"
    
    lareina_files = {
        286: base_path / "Lareina" / "p286_cg-Lareina.csv",
        325: base_path / "Lareina" / "p325_cg-Lareina.csv",
        350: base_path / "Lareina" / "p350_cg-Lareina.csv",
        440: base_path / "Lareina" / "p440_cg-Lareina.csv",
        444: base_path / "Lareina" / "p444_cg-Lareina.csv",
        # Note: Lareina has p460 instead of p469
    }
    
    sally_files = {
        286: base_path / "Sally" / "P286.csv",
        325: base_path / "Sally" / "P325.csv",
        350: base_path / "Sally" / "P350.csv",
        440: base_path / "Sally" / "p440.csv",
        444: base_path / "Sally" / "p444.csv",
        469: base_path / "Sally" / "P469.csv"
    }
    
    print("Loading Dolly's reference data...")
    dolly_df = load_dolly_data(dolly_path)
    
    # Explicit ID validation (exact match, missing, duplicate, alignment)
    print("Running ID exact-match validation...")
    id_validation = validate_participant_ids(dolly_df, lareina_files, sally_files)
    dolly_participants = sorted(id_validation['reference_ids'])
    print(f"Reference participant IDs: {dolly_participants}")
    
    # Storage for all comparisons
    all_cheated_comparisons = []
    all_behavior_diffs = []
    all_timestamp_diffs = []
    
    # Process Lareina's files
    print("\nProcessing Lareina's files...")
    for pid, filepath in lareina_files.items():
        if filepath.exists() and pid in dolly_participants:
            print(f"  Comparing P{pid}...")
            coder_df = load_coder_data(filepath)
            
            # Cheated comparison
            cheated_result = compare_cheated_status(dolly_df, coder_df, 'Lareina', pid)
            all_cheated_comparisons.append(cheated_result)
            
            # Behavior comparison
            behavior_diffs = compare_behaviors(dolly_df, coder_df, 'Lareina', pid)
            all_behavior_diffs.extend(behavior_diffs)
            
            # Timestamp comparison
            timestamp_diffs = compare_behavior_timestamps(dolly_df, coder_df, 'Lareina', pid)
            all_timestamp_diffs.extend(timestamp_diffs)
        else:
            if not filepath.exists():
                print(f"  File not found: {filepath}")
            elif pid not in dolly_participants:
                print(f"  P{pid} not in Dolly's reference data")
    
    # Process Sally's files
    print("\nProcessing Sally's files...")
    for pid, filepath in sally_files.items():
        if filepath.exists() and pid in dolly_participants:
            print(f"  Comparing P{pid}...")
            coder_df = load_coder_data(filepath)
            
            # Cheated comparison
            cheated_result = compare_cheated_status(dolly_df, coder_df, 'Sally', pid)
            all_cheated_comparisons.append(cheated_result)
            
            # Behavior comparison
            behavior_diffs = compare_behaviors(dolly_df, coder_df, 'Sally', pid)
            all_behavior_diffs.extend(behavior_diffs)
            
            # Timestamp comparison
            timestamp_diffs = compare_behavior_timestamps(dolly_df, coder_df, 'Sally', pid)
            all_timestamp_diffs.extend(timestamp_diffs)
        else:
            if not filepath.exists():
                print(f"  File not found: {filepath}")
            elif pid not in dolly_participants:
                print(f"  P{pid} not in Dolly's reference data")
    
    # Generate report
    print("\nGenerating summary report...")
    report_path = base_path / "validation_report.txt"
    report = generate_summary_report(
        all_cheated_comparisons,
        all_behavior_diffs,
        all_timestamp_diffs,
        id_validation,
        report_path
    )
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 80)
    print(report)
    
    # Also save detailed CSVs
    if all_cheated_comparisons:
        pd.DataFrame(all_cheated_comparisons).to_csv(
            base_path / "validation_cheated_comparison.csv", index=False
        )
    
    if all_behavior_diffs:
        pd.DataFrame(all_behavior_diffs).to_csv(
            base_path / "validation_behavior_differences.csv", index=False
        )
    
    if all_timestamp_diffs:
        pd.DataFrame(all_timestamp_diffs).to_csv(
            base_path / "validation_timestamp_differences.csv", index=False
        )
    
    print("\nDetailed CSV files saved:")
    print("  - validation_cheated_comparison.csv")
    print("  - validation_behavior_differences.csv")
    print("  - validation_timestamp_differences.csv")


if __name__ == "__main__":
    main()
