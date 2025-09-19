import re
import sys
import os
import pandas as pd


INPUT_DATASET  = r"C:\Users\natha\Datasets\BI_NEW200.csv"   # <-- set your input CSV path here
OUTPUT_DATASET = r"BI_AHI3_results.csv"                    # <-- set your default output CSV path here

# Encoding fallback behavior when reading CSV
PRIMARY_ENCODING   = "utf-8"
FALLBACK_ENCODING  = "latin-1"

EXCLUSION_KEYWORDS = ["rem", "supine", "non-supine", "prone", "lateral", "residual"]

# -------------------------
# Regex libraries (RDI REMOVED)
# -------------------------

# AHI3% prioritized patterns
AHI3_PATTERNS = [
    re.compile(r'\bAHI3%\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bpAHI3%\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bTotal\s+AHI3%\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI3%\s*=\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI3%\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s*3%\s+(?P<val>\d+(?:\.\d+)?)(?:/hr)?', re.IGNORECASE),
    re.compile(r'\bAHI\s*\(3%\)\s+(?P<val>\d+(?:\.\d+)?)(?:/hr)?', re.IGNORECASE),
    # Side-by-side formats (capture the 3% value):
    # "AHI 3% 10 /4% 8"
    re.compile(r'\bAHI\s*3%\s+(?P<val>\d+(?:\.\d+)?)\s*/\s*4%\s*\d+(?:\.\d+)?', re.IGNORECASE),
    # "AHI 4% 8 /3% 10"
    re.compile(r'\bAHI\s*4%\s*\d+(?:\.\d+)?\s*/\s*3%\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s*3%\s+of\s+(?P<val>\d+(?:\.\d+)?)(?:/hour)?', re.IGNORECASE),
]

# General AHI patterns (unchanged)
AHI_PATTERNS = [
    re.compile(r'\bAHI\s*:\s*\(total\)\s*::\s*(?P<val>\d+(?:\.\d+)?)\s*events/hour', re.IGNORECASE),
    re.compile(r'\bAHI\s*:\s*\(total\)\s*(?P<val>\d+(?:\.\d+)?)\s*events/hour', re.IGNORECASE),
    re.compile(r'\bOverall\s+AHI\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*per\s*hour', re.IGNORECASE),
    re.compile(r'\bOverall\s+AHI\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bTotal\s+AHI\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*per\s*hour', re.IGNORECASE),
    re.compile(r'\bTotal\s+AHI\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*per\s*hr', re.IGNORECASE),
    re.compile(r'\bpAHI\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s+of\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bwas\s+(?P<val>\d+(?:\.\d+)?)\s*events/hr', re.IGNORECASE),
    re.compile(r'Apnea-Hypopnea\s+Index\s*\(AHI\)\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*per\s*hour', re.IGNORECASE),
    re.compile(r'Apnea\s*/\s*Hypopnea\s+Index\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI=\s*(?P<val>\d+(?:\.\d+)?)\s*per\b', re.IGNORECASE),
    re.compile(r'\bapnea-hypopnea\s+index\s+was\s+(?P<val>\d+(?:\.\d+)?)\s*per\s*hour', re.IGNORECASE),
    re.compile(r'\bAHI\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*events\s*/\s*hour', re.IGNORECASE),
    re.compile(r'Apnea/Hypopnea\s+Index\s*\(AHI\)\s+of\s+(?P<val>\d+(?:\.\d+)?)\s*events/hour', re.IGNORECASE),
    re.compile(r'\(AHI\)\s+of\s+(?P<val>\d+(?:\.\d+)?)\s*events\s+per\s+hour', re.IGNORECASE),
    re.compile(r'\(AHI\)\s+of\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bPSG\s+AHI\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bTotal\s+AHI\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    # Guard against AHI3%/AHI4% being mis-captured as plain AHI:
    re.compile(r'\bAHI\s+(?!\d+%)(?P<val>\d+(?:\.\d+)?)\b', re.IGNORECASE),
    re.compile(r'\bAHI\s*[–—-]\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s+was\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s+is\s+(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bAHI\s*\(apnea[-\s]?hypopnea index\)\s*,?\s*events per hour of sleep\s*:\s*(?P<val>\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'Apnea-?Hypopnnea\s+Index\s*\(AHI\)\s*:\s*(?P<val>\d+(?:\.\d+)?)\s*per\s*hour', re.IGNORECASE),
]

# ESS patterns (unchanged)
ESS_PATTERNS = [
    re.compile(r'Epworth\s+Sleepiness\s+Scale\s*:\s*(?P<val>\d+)\s*out\s+of\s+24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Scale\s*:\s*(?P<val>\d+)\s*/\s*24', re.IGNORECASE),
    re.compile(r'EPWORTH\s+SLEEPINESS\s+SCALE\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Scale\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'ESS\s+of\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Score\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s*:\s*(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'EPWORTH\s*:\s*(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'EPWORTH\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s*\(ESS\)\s*:\s*(?P<val>\d+)\s*out\s+of\s+a\s+possible\s+24', re.IGNORECASE),
    re.compile(r'Epworth\s+(?P<val>\d+)\b', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Scale\s*\(0-24\)\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Scale\s+was\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s+of\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s+is\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+is\s+(?P<val>\d+)\s+out\s+of\s+24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleep\s+score\s*:?\s*(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+was\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+is\s+only\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+score\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s*=\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+score\s+is\s+estimated\s+at\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+scale\s+only\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+score\s+is\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+score=\s*(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s*:\s*(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+is\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+was\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s+today\s+is\s+(?P<val>\d+)\s*out\s+of\s+24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+quite\s+high,\s*(?P<val>\d+)\s*out\s+of\s+24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s+today\s+is\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+is\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleep\s+score\s*:\s*(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+of\s+(?P<val>\d+)', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+last\s+visit\s+was\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+was\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+Sleepiness\s+Score\s+today\s+is\s+(?P<val>\d+)/24\.', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+scale\s+is\s+(?P<val>\d+)/24', re.IGNORECASE),
    re.compile(r'Epworth\s+sleepiness\s+score\s+is\s+(?P<val>\d+)', re.IGNORECASE),
]

# Optional: treat two numbers within <= tol as the same.
DEDUPE_TOLERANCE = 0.0  # set to e.g. 0.1 if you want near-equals merged

def dedupe_preserve_order(seq, tol: float = DEDUPE_TOLERANCE):
    """Return seq with duplicates removed, preserving first-seen order."""
    out = []
    if tol <= 0:
        seen = set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
    else:
        for x in seq:
            if not any(abs(x - y) <= tol for y in out):
                out.append(x)
    return out


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Try UTF-8 first; if it fails, fall back to latin-1 (common for hospital exports)."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def preceding_contains_exclusion(text: str, match_start: int) -> bool:
    start_index = max(0, match_start - 20)
    preceding = text[start_index:match_start].lower()
    return any(keyword in preceding for keyword in EXCLUSION_KEYWORDS)


# -------------------------
# Collect *all* AHI/3% matches in textual order (RDI REMOVED)
# -------------------------
def extract_all_ahi_matches(narrative: str):
    """
    Returns a list of dicts:
        [{'kind': 'AHI'|'AHI3', 'value': float, 'phrase': str, 'start': int}, ...]
    in textual order. Honors EXCLUSION_KEYWORDS using a 20-char lookback.
    """
    if not isinstance(narrative, str):
        narrative = str(narrative) if narrative is not None else ""

    pattern_specs = []
    for pat in AHI3_PATTERNS:
        pattern_specs.append(('AHI3', pat))
    for pat in AHI_PATTERNS:
        pattern_specs.append(('AHI', pat))

    matches = []
    seen_spans = set()

    for kind, pat in pattern_specs:
        for m in pat.finditer(narrative):
            if 'val' not in m.groupdict():
                continue
            start_val = m.start('val')
            if preceding_contains_exclusion(narrative, start_val):
                continue
            span = (m.start(), m.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            try:
                val = float(m.group('val'))
            except ValueError:
                continue
            matches.append({
                'kind': kind,
                'value': val,
                'phrase': m.group(0).strip(),
                'start': start_val
            })

    matches.sort(key=lambda d: d['start'])  # textual order
    return matches


def extract_ess(narrative: str):
    if not isinstance(narrative, str):
        narrative = str(narrative) if narrative is not None else ""
    phrases = []
    value = None
    for pat in ESS_PATTERNS:
        for m in pat.finditer(narrative):
            if 'val' not in m.groupdict():
                continue
            phrase = m.group(0).strip()
            if phrase not in phrases:
                phrases.append(phrase)
            if value is None:
                try:
                    value = int(m.group('val'))
                except ValueError:
                    pass
        if value is not None:
            break
    return value, phrases


def process_file(input_path: str, output_path: str = "KPSC_TOTAL.csv"):
    if not os.path.exists(input_path):
        print(f"ERROR: Input file '{input_path}' not found.")
        sys.exit(1)
    # Prevent accidental self-overwrite
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        print("ERROR: Output path would overwrite input file. Aborting.")
        sys.exit(1)

    try:
        df = safe_read_csv(input_path, dtype=str)
    except Exception as e:
        print(f"ERROR: Failed to read CSV file. {e}")
        sys.exit(1)

    required_cols = {'MRN', 'PROC_DATE', 'CODE', 'NARRATIVE'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"ERROR: Missing required columns: {', '.join(sorted(missing))}")
        sys.exit(1)

    df['NARRATIVE'] = df['NARRATIVE'].astype(str).fillna('')
    df['_ROW_POS'] = range(len(df))  # preserve original row order

    # Row-level extraction
    ahi_matches_col, ess_values, ess_phrases = [], [], []
    for _, row in df.iterrows():
        matches = extract_all_ahi_matches(row['NARRATIVE'])
        for d in matches:
            d['row_pos'] = row['_ROW_POS']
        ahi_matches_col.append(matches)

        val_ess, phrases_ess = extract_ess(row['NARRATIVE'])
        ess_values.append(val_ess)
        ess_phrases.append(phrases_ess)

    df['AHI_matches'] = ahi_matches_col
    df['ESS_value']   = ess_values
    df['ESS_phrases'] = ess_phrases

    def aggregate_group(g: pd.DataFrame):
        code_first = g['CODE'].iloc[0]

        # ESS
        all_ess_phrases = []
        for lst in g['ESS_phrases']:
            for p in lst:
                if p not in all_ess_phrases:
                    all_ess_phrases.append(p)
        ess_val = next((v for v in g['ESS_value'] if pd.notnull(v)), None)

        # Collate all matches in stable order across rows
        all_matches = []
        for _, r in g.iterrows():
            all_matches.extend(r['AHI_matches'])
        all_matches.sort(key=lambda d: (d['row_pos'], d['start']))

        # Ordered numeric values
        ahi_values  = [d['value'] for d in all_matches if d['kind'] == 'AHI']
        ahi3_values = [d['value'] for d in all_matches if d['kind'] == 'AHI3']

        # De-duplicate within each report (preserve order)
        ahi_values  = dedupe_preserve_order(ahi_values)
        ahi3_values = dedupe_preserve_order(ahi3_values)

        # Phrase audit trail (optional)
        phrase_trail = [d['phrase'] for d in all_matches]

        # Presence flags
        ahi_bool = int(bool(ahi_values) or bool(ahi3_values))  # 1 if any AHI or AHI3% value found
        ess_bool = int(ess_val is not None)                    # 1 if ESS extracted (0 counts as present)

        return pd.Series({
            'CODE': code_first,
            'AHI_vals': ahi_values,     # expanded later -> AHI-1, AHI-2, ...
            'AHI3_vals': ahi3_values,   # expanded later -> AHI3%-1, AHI3%-2, ...
            'NARRATIVE_AHI': '; '.join(phrase_trail) if phrase_trail else None,
            'NARRATIVE_ESS': '; '.join(all_ess_phrases) if all_ess_phrases else None,
            'ESS': ess_val,
            'AHI_bool': ahi_bool,
            'ESS_bool': ess_bool
        })

    result = (
        df.groupby(['MRN', 'PROC_DATE'], sort=False)
          .apply(aggregate_group)
          .reset_index()
    )

    # Expand AHI_vals into AHI-1, AHI-2, ...
    max_len_ahi = result['AHI_vals'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
    max_len_ahi = int(max_len_ahi) if pd.notnull(max_len_ahi) else 0
    for i in range(max_len_ahi):
        col = f'AHI-{i+1}'
        result[col] = result['AHI_vals'].apply(lambda x: (x[i] if isinstance(x, list) and len(x) > i else None))

    # Expand AHI3_vals into AHI3%-1, AHI3%-2, ...
    max_len_ahi3 = result['AHI3_vals'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
    max_len_ahi3 = int(max_len_ahi3) if pd.notnull(max_len_ahi3) else 0
    for j in range(max_len_ahi3):
        col = f'AHI3%-{j+1}'
        result[col] = result['AHI3_vals'].apply(lambda x: (x[j] if isinstance(x, list) and len(x) > j else None))

    # Final column order (note: no 'Total AHI')
    cols_base = ['MRN', 'PROC_DATE', 'CODE']
    cols_ahi  = [f'AHI-{i+1}' for i in range(max_len_ahi)]
    cols_ahi3 = [f'AHI3%-{j+1}' for j in range(max_len_ahi3)]
    cols_tail = ['ESS', 'AHI_bool', 'ESS_bool', 'NARRATIVE_AHI', 'NARRATIVE_ESS']

    keep_cols = [c for c in cols_base + cols_ahi + cols_ahi3 + cols_tail if c in result.columns]
    result = result[keep_cols]

    # Clean up temp lists
    result = result.drop(columns=['AHI_vals', 'AHI3_vals'], errors='ignore')

    # Write
    try:
        result.to_csv(output_path, index=False)
    except Exception as e:
        print(f"ERROR: Failed to write output CSV. {e}")
        sys.exit(1)

    print(f"Processing complete. Output written to '{output_path}'.")


def resolve_paths_from_cli_or_defaults():
    """
    Returns (input_path, output_path)
    Priority:
      1) If CLI args provided: use them (argv[1] and optional argv[2])
      2) Otherwise, use INPUT_DATASET and OUTPUT_DATASET variables
    """
    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_DATASET
        return in_path, out_path
    else:
        return INPUT_DATASET, OUTPUT_DATASET


def main():
    input_path, output_path = resolve_paths_from_cli_or_defaults()
    if not input_path or input_path.strip() == "":
        print("ERROR: No input path provided. Set INPUT_DATASET or pass a path via CLI.")
        sys.exit(1)
    process_file(input_path, output_path)


if __name__ == "__main__":
    main()
