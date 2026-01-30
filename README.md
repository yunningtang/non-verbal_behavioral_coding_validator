# non-verbal_behavioral_coding_validator
A Python pipeline for validating inter-rater reliability in behavioral coding studies with timestamp matching. 
# Behavioral Coding Validator

A Python pipeline for validating inter-rater reliability in behavioral coding studies. Compares multiple coders' outputs against a reference coder using timestamp matching, IoU computation, and statistical analysis.

## Key Features

**ID Validation** - Exact match verification of participant identifiers  
**Behavioral Status Comparison** - Binary classification agreement (e.g., cheated/not cheated)  
**Behavior Presence Check** - Identifies missing or extra coded behaviors  
**Timestamp Alignment** - Nearest-neighbor matching with configurable thresholds  
**IoU Computation** - Intersection over Union for segment overlap quality（please ignore for now) 
**Comprehensive Reporting** - Precision, recall, F1 scores, and detailed discrepancy logs  
**Visual Analytics** - Automated generation of distribution plots and bias analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/behavioral-coding-validator.git
cd behavioral-coding-validator

# Install dependencies
pip install pandas matplotlib seaborn numpy
```

### Basic Usage

```bash
# Run the validation pipeline
python validation_pipeline.py

# Generate enhanced analysis with visualizations
python enhanced_validation_analysis.py
```

### Input Structure

```
project/
├── Dolly/
│   └── practice coding.csv         # Reference coder
├── Lareina/
│   ├── p286_cg-Lareina.csv         # Coder 1 files
│   ├── p325_cg-Lareina.csv
│   └── ...
└── Sally/
    ├── P286.csv                    # Coder 2 files
    ├── P325.csv
    └── ...
```

### Required CSV Columns

- `ID` - Participant identifier
- `Cheated or not` - Binary behavioral classification
- `Behavior` - Behavior type label
- `Start (s)` - Segment start time in seconds
- `Stop (s)` - Segment stop time in seconds
- `Image index start` - Frame number at segment start (optional)
- `Image index stop` - Frame number at segment end (optional)

## Output Files

| File | Description |
|------|-------------|
| `validation_report.txt` | Comprehensive text report with all findings |
| `validation_cheated_comparison.csv` | Binary classification agreement results |
| `validation_behavior_differences.csv` | Missing/extra behavior discrepancies |
| `validation_timestamp_differences.csv` | Detailed timestamp alignment data |
| `enhanced_validation_report.txt` | Statistical summary with recommendations |
| `validation_analysis_*.png` | Visualization plots |
| `behavior_quality_analysis.csv` | Per-behavior quality metrics |

## Methodology

### Matching Algorithm

The pipeline uses **nearest-neighbor matching** by segment start time:

1. For each reference segment, find the closest coder segment of the same behavior
2. Pair segments only if `|start_time_diff| ≤ MATCH_THRESHOLD` (default: 10s)
3. Each coder segment is used at most once (greedy matching)
4. Compute IoU for overlap quality assessment

### Key Parameters

```python
MATCH_THRESHOLD_SEC = 10.0      # Maximum start time difference for pairing
IOU_THRESHOLD = 0.5             # Minimum IoU for good overlap
SUMMARY_MAX_START_DIFF_SEC = 30.0  # Outlier exclusion for statistics
```

### Interpretation Guide

- **FN (False Negative)**: Reference has a segment, no coder match within threshold → Coder missed this behavior
- **FP (False Positive)**: Coder has a segment, no reference match within threshold → Coder coded extra behavior
- **IoU = 0**: Segments paired by start time but no temporal overlap (common for instantaneous events) （please ignore for now) 
- **IoU > 0.5**: Good segment alignment (please ignore for now)      

## Example Results

```
EXECUTIVE SUMMARY
Reference contains 6 participant IDs.
Cheated-or-not agreement: 10/11 comparisons match reference.
Behavior timestamps: 200 segments paired; 15 with IoU=0; 5 FN; 8 FP.

SECTION 4: SUMMARY STATISTICS BY CODER


## Customization

### Adjust Matching Threshold

Edit `validation_pipeline.py`:

```python
MATCH_THRESHOLD_SEC = 15.0  # Increase for more lenient matching
```

### Add New Coders

Update file paths in `main()`:

```python
new_coder_files = {
    286: base_path / "NewCoder" / "p286.csv",
    325: base_path / "NewCoder" / "p325.csv",
    # ...
}
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
