.venv/bin/python src/cli.py compare-t1-runs \
  --run-dirs \
  results/milestones/v2026.09/t1.1-C0.1-2026-10-29 \
  results/milestones/v2026.09/t1.1-C1-2026-10-27 \
  results/milestones/v2026.09/t1.1-C10-2026-10-29
T1 comparison: OK (3 sources, 3 pairs)

Sources
  1  /data/all/projects/git/vepkar-morph-disambig/results/milestones/v2026.09/t1.1-C0.1-2026-10-29  t1.1  C=0.1
  2  /data/all/projects/git/vepkar-morph-disambig/results/milestones/v2026.09/t1.1-C1-2026-10-27  t1.1  C=1
  3  /data/all/projects/git/vepkar-morph-disambig/results/milestones/v2026.09/t1.1-C10-2026-10-29  t1.1  C=10

Pairs
Pair     A→B   Reason  Corrected  Regressed  ΔTop-1  ΔMRR  ΔTop-3
1 vs 2     1→2    Top-1 128        61        +0.0237  +0.0124 +0.0004
1 vs 3     1→3    Top-1 180        101       +0.0280  +0.0149 +0.0007
2 vs 3     2→3    Top-1 66         54        +0.0042  +0.0026 +0.0004

Best run: source 3 (C=10) leads all three metrics

Outputs
  1 vs 2  /data/all/projects/git/vepkar-morph-disambig/work/comparisons/v2026.09/t1.1-C0.1__vs__t1.1-C1.csv
  1 vs 3  /data/all/projects/git/vepkar-morph-disambig/work/comparisons/v2026.09/t1.1-C0.1__vs__t1.1-C10.csv
  2 vs 3  /data/all/projects/git/vepkar-morph-disambig/work/comparisons/v2026.09/t1.1-C1__vs__t1.1-C10.csv
