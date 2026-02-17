# Training Status
Last updated: 2026-02-17 14:11 EET

## Training Progress

| Task | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|------|--------|--------|--------|--------|--------|
| Task601 (TotalSegV1, 104 cls) | DONE | RUNNING | DONE | DONE | DONE |
| Task611 (Group1, 13 cls) | RUNNING | not started | RUNNING | RUNNING | not started |
| Task612 (Group2, 25 cls) | DONE | DONE | DONE | DONE | RUNNING |
| Task613 (Group3, 10 cls) | DONE | DONE | RUNNING | RUNNING | RUNNING |
| Task614 (Group4, 60 cls) | RUNNING | RUNNING | RUNNING | RUNNING | RUNNING |

Legend: DONE = final checkpoint saved · RUNNING = job active · not started = no checkpoint

## Current SLURM Jobs (as of 2026-02-17 14:11)

| Job ID | Task | Folds | State | Runtime |
|--------|------|-------|-------|---------|
| 16123663 | Task601 fold 1 | 1 | RUNNING | 1d 15h+ |
| 16149558_4 | Task612 | fold 4 | RUNNING | ~16h |
| 16149559_2,3,4 | Task613 | fold 2,3,4 | RUNNING | 3–5h |
| 16149560_0–4 | Task614 | fold 0–4 | RUNNING | ~3h |
| 16150950_2,3 | Task611 | fold 2,3 | RUNNING | 2–3h |
| 16150950_4 | Task611 | fold 4 | PENDING | — |

## Next Steps
- [ ] Submit Task611 folds 1 and 4 (fold 4 is pending, fold 1 not started)
- [ ] Run validation on Task612 once fold 4 completes (job 16149558_4)
- [ ] Run validation on Task613 once folds 2–4 complete
- [ ] Monitor Task614 (all 5 folds running ~3h each)
- [ ] Monitor Task601 fold 1 (running 1d+, may need to check progress)
