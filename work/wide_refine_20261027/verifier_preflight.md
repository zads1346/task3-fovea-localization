# Wide-window independent verification

The new `verify_results.py` is isolated from all older experiment scripts. Importing it never starts training or CUDA inference. Only its explicit `main()` performs final checkpoint replay, after both full five-fold experiments are complete. A completed fold can be checked independently by `fold_record_audit` without a GPU model.

The frozen protocol is internally consistent: WIDE_DEV and the inherited SINGLE control were fitted on the same development images. Their comparison uses the original held calibration group, now explicitly named `selection_then_final_fit`. The gate requires at least 0.5 px and 2% RMSE improvement, with no P90 increase. The decision is written before FULL refitting. FULL adds this selection group, keeps the separately selected epoch count, and does not fit or reuse an uncertainty or risk calibrator. Outer validation remains excluded from fitting, epoch selection and branch selection.

Checks implemented:

- Frozen source SHA and all protected previous file SHA/size records; previous submission identity and source checkpoint lineage.
- Both fixed group partitions, exact role relabeling, train/test ID and image-hash disjointness, and absence of secondary test inference.
- Fitting-only group crossfit centres; original coarse predictions for the newly added FULL images; exact .25 prediction-centred proposal windows without target corrections.
- All 30 stage signatures, training/stopping memberships, crop operands, finite histories and actual optimizer updates; fixed epoch counts for DEV/FULL.
- Independent coordinate RMSE, Euclidean tail errors, worse-than-coarse fractions, and proposal-window coverage. The .25 coverage is a geometric diagnostic and is not presented as the physical window of a blended model.
- Independent three-condition gate reconstruction and fold blends: accepted primary folds use 0.75 previous + 0.25 WIDE_FULL; rejected folds retain the previous result. All ensemble means, original submission reconstruction, template IDs, coordinate bounds and 40 submitted values are checked.
- Final CUDA replay loads 20 new DEV/FULL checkpoints and 10 inherited SINGLE controls, using each original model configuration and each producer frame's original order and batch partition. Tolerance remains 0.001 px. No two-image subset is compared against outputs produced with a different batch shape.

CSV parsing intentionally differs by provenance: previous source files use the producer's original default pandas parsing; new files use round-trip parsing to recover the serialized training operands exactly. Signature mismatches are investigated rather than bypassed or given numeric tolerance.

Limitations are explicit. The wider treatment changes ROI width, input resolution and heatmap resolution together. DEV versus FULL changes sample count and optimizer-update count. FULL has no independent post-refit calibration group. The second split reuses the same training population and is a stability audit, not a new patient cohort. No official leaderboard or external-reference score is claimed by this verifier, and it never reads such labels.

Preflight status: syntax compilation succeeded. Final PASS will only be written after all record checks and all 30 checkpoint replays actually succeed. Until that execution, this document is a description of checks, not a completed-validation claim.

First completed-fold CPU precheck: split 2026, fold 0 passed all three stage signatures, lineage/role checks, gate arithmetic, saved metrics and blend checks. CUDA remained uninitialized. The observed result is recorded in `primary_record_precheck_0.json`; this is not a GPU-replay or all-fold completion claim.

## Checkpoint-replay compatibility correction

After all 30 stage-record checks passed, the first complete verification attempt stopped at inherited SINGLE OOF replay in split 2026, fold 1: maximum difference 0.002197265625 px exceeded the unchanged 0.001 px tolerance. This was a checker input-provenance error. The wide pipeline copies old ROI CSVs using default pandas parsing but inherits the old SINGLE OOF/test predictions without rerunning them. Replaying those inherited predictions must use the previous producer's original ROI operands, read from the protected original ROI CSV with round-trip parsing. Newly produced SINGLE selection predictions must continue using the new selection ROI CSV.

An authorized same-model, same-full-frame, same-batch replay confirmed the cause: the default-parsed copy produced the exact failing 0.002197265625 px difference; the original round-trip ROI produced **0.0 px** difference. Only one ROI side differed (290.09999999999997 versus 290.1), with identical centres. Evidence is in `roi_replay_diagnosis.json`. The verifier now distinguishes those two provenances. No model, training configuration, gate, submission file or tolerance was changed. The first failed attempt remains a failure; the full corrected verifier must run again before a final PASS is claimed.
