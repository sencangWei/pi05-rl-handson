# Joint action choice, bounded comparison

Goal: compare absolute-target baseline with observation-anchored residual targets; assess velocity eligibility without claiming an untested velocity policy. No UI promotion, no hardware.

1. Inspect existing evidence and freeze scope — complete.
2. Isolated residual adapter, tests and expert physical replay — complete (965 controls, max physics error4.16e-11).
3. Matched 200 updates and identical two-state closed-loop evaluation — complete. Absolute0/2, residual0/2; residual holdout600 controls without native goal.
4. Report measured limits and action-contract recommendation — complete. Owned remote backup and independent restore verification pending.

Original sources intact; no holdout gradients/statistics; no limit relaxation, no expert fallback; no unrelated jobs stopped. Absolute baseline is frozen existing evidence, not another seed replication. Store absolute commanded targets; keep position PD execution; residual training remains a candidate, not a validated winner.
