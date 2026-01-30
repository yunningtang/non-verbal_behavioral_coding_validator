"""
Produces behavior_presence_counts.csv: counts per participant, coder, behavior.
Columns: participant_id, coder, behavior, dolly_count, coder_count.
No commentary or metrics; table only.
"""
import pandas as pd
from pathlib import Path

from validation_pipeline import (
    load_dolly_data,
    load_coder_data,
    get_dolly_data_for_participant,
    build_behavior_presence_counts,
)


def main():
    base_path = Path(__file__).parent
    dolly_path = base_path / "Dolly" / "practice coding.csv"
    lareina_files = {
        286: base_path / "Lareina" / "p286_cg-Lareina.csv",
        325: base_path / "Lareina" / "p325_cg-Lareina.csv",
        350: base_path / "Lareina" / "p350_cg-Lareina.csv",
        440: base_path / "Lareina" / "p440_cg-Lareina.csv",
        444: base_path / "Lareina" / "p444_cg-Lareina.csv",
    }
    sally_files = {
        286: base_path / "Sally" / "P286.csv",
        325: base_path / "Sally" / "P325.csv",
        350: base_path / "Sally" / "P350.csv",
        440: base_path / "Sally" / "p440.csv",
        444: base_path / "Sally" / "p444.csv",
        469: base_path / "Sally" / "P469.csv",
    }
    dolly_df = load_dolly_data(dolly_path)
    ref_ids = sorted(dolly_df["ID"].dropna().astype(int).unique())
    rows = build_behavior_presence_counts(dolly_df, lareina_files, sally_files, ref_ids)
    out_path = base_path / "behavior_presence_counts.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
