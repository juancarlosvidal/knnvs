"""
make_table.py
-------------
Reads lar1.csv (with feature selection) and lar1_no.csv (without feature
selection), computes the mean MSE over the 300 Monte Carlo simulations for
every (N, p) combination, and prints the LaTeX table that matches tab:lar1.

Usage:
    python make_table.py [--fs lar1.csv] [--nofs lar1_no.csv]

Column names expected in each CSV:
    DESC, N_SIMS, ACC_FS_A, PVA, ACC_FS_V, PVV, N, P, KA, KV, MSEA, MSEV
"""

import argparse
import csv
import sys
from collections import defaultdict

# ── configuration ─────────────────────────────────────────────────────────────
N_LIST = [5000, 10000, 20000, 50000, 100000]
P_LIST = [10, 25, 50, 100]
DECIMALS = 4          # digits after the decimal point in the table


# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path: str) -> dict:
    """Return {(N, P): {'msea': [floats], 'msev': [floats]}}."""
    data: dict = defaultdict(lambda: {"msea": [], "msev": []})
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (int(row["N"]), int(row["P"]))
            data[key]["msea"].append(float(row["MSEA"]))
            data[key]["msev"].append(float(row["MSEV"]))
    return data


def mean(values: list) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float) -> str:
    return f"{value:.{DECIMALS}f}"


# ── LaTeX generation ──────────────────────────────────────────────────────────
def build_table(fs_data: dict, nofs_data: dict) -> str:
    lines = []

    header = r"""\begin{table*}[!t]
	\caption{Performance evaluations in Scenario~8 (large regime) of the estimators $\overline{x}$ and $\hat{v}$ for the different VS-$k$-NN settings described in the text, with and without feature selection. We report the mean squared error, averaging over 300 Monte Carlo simulations.}
	\label{tab:lar1}
	\centering
	\begin{tabular}{c|c|cc|cc|cc|cc}
		\multicolumn{2}{c}{$\;$} & \multicolumn{2}{c}{$p=10$} & \multicolumn{2}{c}{$p=25$} & \multicolumn{2}{c}{$p=50$} & \multicolumn{2}{c}{$p=100$}  \\
		& $N$ & $\overline{x}$ & $\hat{v}$ & $\overline{x}$ & $\hat{v}$ & $\overline{x}$ & $\hat{v}$ & $\overline{x}$ & $\hat{v}$ \\
		\hline"""
    lines.append(header)

    def data_rows(data: dict, label: str) -> list:
        rows = []
        n_rows = len(N_LIST)
        for idx, n in enumerate(N_LIST):
            cells = []
            for p in P_LIST:
                key = (n, p)
                msea = mean(data[key]["msea"]) if key in data else float("nan")
                msev = mean(data[key]["msev"]) if key in data else float("nan")
                cells.append(f"{fmt(msea)} & {fmt(msev)}")
            row_data = " & ".join(cells)
            n_fmt = f"{n:,}".replace(",", r"\,")   # thin space thousands sep
            if idx == 0:
                prefix = (
                    rf"		\parbox[t]{{2mm}}{{\multirow{{{n_rows}}}{{*}}"
                    rf"{{\rotatebox[origin=c]{{90}}{{{label}}}}}}} & "
                )
            else:
                prefix = "        & "
            rows.append(f"{prefix}{n_fmt} & {row_data} \\\\")
        return rows

    # FS block
    for row in data_rows(fs_data, "FS"):
        lines.append(row)
    lines.append(r"		\hline")

    # No FS block
    for row in data_rows(nofs_data, "No FS"):
        lines.append(row)
    lines.append(r"		\hline")

    footer = r"""	\end{tabular}
\end{table*}"""
    lines.append(footer)

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from simulation CSVs.")
    parser.add_argument("--fs",   default="lar1.csv",    help="CSV with feature selection (default: lar1.csv)")
    parser.add_argument("--nofs", default="lar1_no.csv", help="CSV without feature selection (default: lar1_no.csv)")
    parser.add_argument("--out",  default=None,          help="Output .tex file (default: print to stdout)")
    args = parser.parse_args()

    try:
        fs_data = read_csv(args.fs)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.fs}")

    try:
        nofs_data = read_csv(args.nofs)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.nofs}")

    # Quick sanity check
    for label, data in [("FS", fs_data), ("No FS", nofs_data)]:
        for n in N_LIST:
            for p in P_LIST:
                key = (n, p)
                count = len(data.get(key, {}).get("msea", []))
                if count == 0:
                    print(f"WARNING [{label}]: no rows found for N={n}, p={p}", file=sys.stderr)
                elif count != 300:
                    print(f"WARNING [{label}]: expected 300 rows for N={n}, p={p}, got {count}", file=sys.stderr)

    table = build_table(fs_data, nofs_data)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(table + "\n")
        print(f"Table written to {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()