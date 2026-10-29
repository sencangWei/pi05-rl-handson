# Progress

Inspected original joint pilot and controller. New independent branch experiment/joint-residual-choice-20261029. GPU 31574MiB free at discovery; no jobs stopped. Original tracked working tree clean; pre-existing untracked environment files and pair parquet remain untouched.

Implemented isolated residual encoder/dataset/server adapter and physics replay gate; 51 targeted tests passed, Ruff passed. Baseline source files unchanged. First replay source511 controls passed per-control physics equality within tolerance; second source in progress.

Both expert replays completed successfully. Residual training completed200updates with exact gradient sample order; held-out gradient samples0. Actual closed-loop eval: train0controls (J4 limit), holdout600controls (budget, goal0). Stored real videos, state arrays and traces. Export reconstructed71parameter tensors locally; failed experimental weights remain local. Recommendation/report completed; backing up exact new files and validating restore next.

Payload e076537 pushed to owned public remote experiment/joint-residual-choice-20261029. Independent remote Git tree/blob restore matched all3850 tracked files and rebuilt tree4d4b2db8711cf434c49c8d50e2bc804456b91a37. Restored51targetedtests passed. Only this trial service stopped; port9878 confirmed closed. Failed trial checkpoint/adapter remains local, explicitly excluded from remote backup claim.
