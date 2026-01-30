"""
Produces id_exact_match.csv: alignment results for participant ID exact match.
Columns: participant_id, coder, in_reference, in_coder_file, alignment_status.
No commentary or metrics; table only.
"""
import pandas as pd
from pathlib import Path

from validation_pipeline import (
    load_dolly_data,
    load_coder_data,
    validate_participant_ids,
    build_id_exact_match_table,
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
    id_validation = validate_participant_ids(dolly_df, lareina_files, sally_files)
    rows = build_id_exact_match_table(id_validation)
    out_path = base_path / "id_exact_match.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
