import os
import csv


def resume_csv(outfile, header, n_sims):
    """Figure out how much of a sim_*.py grid run is already in outfile.

    Each (n, p) combo in the grid writes exactly n_sims rows in order, so the
    number of valid rows already on disk maps directly to a (combo, i)
    position to resume from. Rows are only counted if they match the header
    width, so a row left truncated by a killed job is dropped rather than
    counted as done.

    Returns (mode, combos_done, i_done, n_done):
      mode        - 'a' to append to a partial file, 'w' to start fresh
      combos_done - number of (n, p) combos already fully written
      i_done      - simulations already done within the first unfinished combo
      n_done      - total rows already present (0 if starting fresh)
    """
    n_done = 0
    if os.path.exists(outfile):
        with open(outfile, newline='') as f:
            rows = list(csv.reader(f))
        if rows and rows[0] == header:
            n_done = sum(1 for r in rows[1:] if len(r) == len(header))

    combos_done, i_done = divmod(n_done, n_sims)
    mode = 'a' if n_done > 0 else 'w'
    return mode, combos_done, i_done, n_done
