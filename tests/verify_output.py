"""Compare a fresh run with the reference run shipped in this package.

    python tests/verify_output.py            # after `python run_all.py`
    python tests/verify_output.py --write    # record a new reference (authors only)

The reference (tests/reference_output.json) holds, for each simulation CSV, its MD5
checksum, row count and the sum of every numeric column, from a run with seed 0 and one
replication. A run matches when the row counts are equal and every column sum agrees to
within 1e-9 relative error; identical checksums mean the files are identical byte for
byte (checksums can differ across pandas versions even when the numbers agree).
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "output", "results")
REFERENCE = os.path.join(ROOT, "tests", "reference_output.json")
FILES = ["modular_simulation_results.csv", "sequential_simulation_results_ath.csv",
         "simulation_results_hta.csv"]


def fingerprint(path):
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    df = pd.read_csv(path)
    sums = {c: float(df[c].sum()) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return {"md5": md5, "rows": int(len(df)), "column_sums": sums}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--write", action="store_true", help="record the current output as the reference")
    args = p.parse_args()
    missing = [f for f in FILES if not os.path.exists(os.path.join(RESULTS, f))]
    if missing:
        sys.exit(f"Missing {missing}; run `python run_all.py` first.")
    current = {f: fingerprint(os.path.join(RESULTS, f)) for f in FILES}
    if args.write:
        json.dump(current, open(REFERENCE, "w"), indent=2)
        print(f"Wrote {os.path.relpath(REFERENCE, ROOT)}")
        return
    reference = json.load(open(REFERENCE))
    ok = True
    for f in FILES:
        cur, ref = current[f], reference[f]
        same_rows = cur["rows"] == ref["rows"]
        bad = [c for c, v in ref["column_sums"].items()
               if abs(cur["column_sums"].get(c, float("nan")) - v) > 1e-9 * max(1.0, abs(v))]
        match = same_rows and not bad
        ok &= match
        status = "MATCHES (byte-identical)" if match and cur["md5"] == ref["md5"] else \
                 "MATCHES (same numbers)" if match else f"DIFFERS (rows ok={same_rows}, columns {bad})"
        print(f"{f}: {status}")
    print("\nOutput matches the reference run." if ok else
          "\nOutput differs from the reference run (expected only with a seed other than 0 or more than one replication).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
