# PromptNLP

PromptNLP is a novel approach that leverages large language models (LLMs) to efficiently develop accurate regular expression (regex) code for extracting clinical metrics from unstructured sleep medicine notes. The tool targets two key values:

- **Apnea-Hypopnea Index (AHI)** — including general AHI values and desaturation-specific variants (AHI 3%, AHI 4%)
- **Epworth Sleepiness Scale (ESS)**

The regex patterns were iteratively refined using LLM-assisted prompt engineering and validated against manually annotated clinical notes from three institutions.

## Scripts

| Script | Description |
|---|---|
| `code3t.py` | Best-performing general extractor. Captures AHI (general), AHI 4%, and ESS from clinical narratives (sleep study reports and free-text clinic notes). |
| `AHI3code.py` | Specialized extractor for AHI 3% values, a format intentionally excluded from Code 3T. |

## Requirements

- Python 3.8+
- pandas >= 1.5.0

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input Format

Both scripts expect a CSV file with the following required columns:

| Column | Description |
|---|---|
| `MRN` | Medical record number (patient identifier) |
| `PROC_DATE` | Procedure/encounter date |
| `CODE` | Procedure or encounter code |
| `NARRATIVE` | Free-text clinical note content |

The CSV may use UTF-8 or Latin-1 encoding (auto-detected).

## Usage

### Command-line

```bash
# Specify input and output paths as arguments
python code3t.py path/to/input.csv path/to/output.csv

# Specify input only (output defaults to code3t_results.csv)
python code3t.py path/to/input.csv

# Same usage for AHI 3% extraction
python AHI3code.py path/to/input.csv path/to/output.csv
```

### Setting default paths

Alternatively, edit the `INPUT_DATASET` and `OUTPUT_DATASET` variables at the top of each script to set default file paths, then run without arguments:

```bash
python code3t.py
```

## Output Format

Each script produces a CSV grouped by `MRN` and `PROC_DATE` with the following columns:

| Column | Description |
|---|---|
| `MRN` | Medical record number |
| `PROC_DATE` | Procedure date |
| `CODE` | Encounter code (first per group) |
| `AHI-1`, `AHI-2`, ... | Extracted general AHI values (in order of appearance) |
| `AHI4%-1`, `AHI4%-2`, ... | Extracted AHI 4% values (`code3t.py`) |
| `AHI3%-1`, `AHI3%-2`, ... | Extracted AHI 3% values (`AHI3code.py`) |
| `ESS` | Extracted Epworth Sleepiness Scale score |
| `AHI_bool` | 1 if any AHI value was found, 0 otherwise |
| `ESS_bool` | 1 if an ESS value was found, 0 otherwise |
| `NARRATIVE_AHI` | Matched AHI phrases (audit trail) |
| `NARRATIVE_ESS` | Matched ESS phrases (audit trail) |

When multiple rows share the same `MRN` and `PROC_DATE`, they are aggregated into a single output row with values deduplicated in order of appearance.

## Exclusion Logic

Matches preceded (within 20 characters) by positional or state qualifiers are excluded to avoid capturing position-specific AHI values rather than overall totals. Excluded keywords: `rem`, `supine`, `non-supine`, `prone`, `lateral`, `residual`.

## License

See [LICENSE](LICENSE) for details.

## Citation

If you use PromptNLP in your research, please cite the associated manuscript (SLEEP Advances, 2026).
