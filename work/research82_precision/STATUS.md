# Precision follow-up status

## COMPLETE AND VERIFIED

Formal training session 47310 exited 0. Five folds and all 15 new stages completed in 576.536 seconds. Only zero-based fold 2 (user-facing fold 3) passed the fixed gate. Baseline OOF RMSE 28.7623882570; narrow single branch 24.0819763355; predeclared guarded submission 28.6730046273. The latter is the delivered candidate, without post-hoc change to gate or blend.

Independent GPU verifier session 58389 exited 0: PASS, 25 final networks reloaded, maximum candidate CSV difference 4.5898423e-7 px, zero test images in training/selection, all 20 baseline networks unchanged. New submission SHA256 a417f0282fb45222d8c05ae8265074b6534a2fb0f6e7b3117323f4f63e64112f; baseline be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f remains intact. All 40 coordinate values changed; mean test prediction displacement 0.580423 px, maximum 4.057807 px. No new Kaggle score, no automatic upload.

finalize.py completed successfully. Output report.md, comparison.json, per-image comparisons, both CSVs, native Notebook, error diagnosis, GPU and CPU audit records, and research82_precision_documents.zip are under outputs/research82_precision_20261027. The 78-file document archive was verified byte-for-byte against its manifest; it excludes model weights. Native Notebook has 95 training stages when rerun from scratch, accepts existing 3pretrained ResNet18 weights, and passed static plus independent CPU orchestration/data/weight checks (not an actual Kaggle execution). Pretrained ZIP remains outputs/fovea_pretrained_weights.zip. The initial phase's broader R18+EfficientNet weight ZIP also remains available.

OOF diagnosis: images 49/72 account for 48.43% of baseline squared error; both component ROI sets miss them and all OOF windows hit minimum fraction .15. Existing fallback cannot detect these cases. This suggests future context/coverage/risk-feature experiments; no hand corrections or added experiments were made from this diagnosis. Deliver the verified candidate as a modest local improvement only; not a proven improvement over Kaggle 8.2. Earlier running-state notes below are archival.

2026-10-27: formal GPU run started through unified session 47310, command `C:/Users/lenovo/anaconda3/envs/dome/python.exe -u work/research82_precision/run.py`, redirected to training.log. Do not launch another training process. Engine is frozen with SHA256 e150f3cc7e164bedabe900ca9599e3d4557b835c1ae379261cfbdbed4a9d6200.

The functional GPU probe passed in 3.45 seconds, 6 optimizer updates, 402 MB allocated peak, official ResNet18 weights verified and encoder weights changed. Its separate probe checkpoint is never used by the formal experiment.

One new hypothesis only: fine heatmap sigma 1.5 instead of 2.5, ResNet18, 256 input, 64 heatmap. Five folds x (fixed selection + multiscale selection + final refit) = 15 new stages. All other source settings and 2% AND 0.5px inner gates unchanged; accepted branch contributes 20%, otherwise original 8.2 baseline. Baseline SHA be3cba8f60f96325c7730480db2f740522e79ac550368984174a73396342176f.

Next: wait for training success, run verify_results.py on GPU (25 final networks), then finalize.py after its independent audit PASS. Native Notebook generated and statically verified; additional independent CPU-only notebook review underway. The document finalizer is being prepared by fovea_papers_82. baseline_audit_82 is writing error_diagnosis.md based only on existing train OOF evidence. Do not tweak rules or run another parameter search after seeing results.

Phase 1 in work/research82 is complete and verified, all gates rejected; its CSV is byte-identical to 8.2. User already informed. Its report and ZIP exist. If phase 2 also rejects all folds, plainly report that tested changes did not improve the baseline and retain it. Never call an unchanged file a new improved prediction. Final should include actual submission, native Notebook, document ZIP, and available ImageNet initialization ZIP; no online score is known for any changed prediction.
