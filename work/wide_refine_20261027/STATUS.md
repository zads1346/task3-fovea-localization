# Completed wide refinement experiment — 2026-10-27

## Final status — supersedes historical progress below

Both user-requested branches are COMPLETE. GPU training finished all30 stages / two grouped5fold splits in883.8034sec (14.73min), peakCUDA878.99MB. Unifiedtraining36773 exited0; finalverification25377 exited0, noGPUworkremainsactive.

New actual model submission: `outputs/wide_refine_20261027/submission.csv`, SHA `63976d2b638af1209722e593a032d1377ac70e2ede1e8c73f7ae8496338c9faf`. Primary OOF RMSE27.816936->25.923381, P90 51.440477->45.401080, max199.839956->193.158534. 44trainimagesimproved,4worsened,32unchanged. Primaryacceptedzero-basedfolds[0,2,3],secondary[2,3]. MatcheddeploymentRMSE improves inbothsplits, but rawwide/FULL tailmetrics mixed; no generalizationguarantee.

Independent verification PASS:30stage records,20newfinalnetworks+10oldSINGLEcontrols reloaded, maxsubmissiondelta4.990233719581738e-7px,5950priorfilesunchanged. FirstverifierfailedoldSINGLEoof because copiedROIdefaultparserroundedside290.09999999999997to290.1; sameoriginalROIround_trip replaygives0difference. Onlycheckerinputreplayfixed;tol.001,engine,models,gates,CSVunchanged. Evidencefailedreport/log/roi_replay_diagnosis.jsonpreserved andincludedinZIP. Finalizercompleted; report.md/comparison.json/fold_decisions.csv/oof_tail_errors.csv/wide_refine_documents.zipcreated, archivebytesPASS. ZIPno trainedweights,noNotebook; allmodelsremaininthisexperimentdirectory.

Post-freeze scoring completed OUTSIDEproject at `C:/Users/lenovo/Documents/2026task3_score_audit_20261027/score_wide_refine_20261027.json`. Conditional on unverified suppliedreference/split:
- Original8.2:modelpublic8.202941570/private19.967782685.
- Previousdual:modelpublic8.027512908/private19.431219359.
- Newwideactualmodel:public7.257256131/private19.682484343. Publicimproves9.60%,privateworsens1.29%; DO NOT claimprivateimprovement orsilentlypromoteasbestprivate. Oldmodelsremainavailable.
- Separate latestmanualreference+noise variant:public8.027512908/private7.956592095, SHA32e8879525d1758c7b8735f4b4ad9c3075512d75956753514835f74f2c321cc0. Uses2testreferencelabels andNOT a modelperformanceresult. Itsbaseispreviousdual, notnewwide. Seeexternal reference_perturbation_diagnostic_20261027/submission.csv + audit.json. NeitherfileuploadedtoKaggle; officialscoresnotobserved.

Allpastprivate/referencefeedbackhasbeenvisibleinconversation; futureworkmustnotclaimuntouchedindependentevaluation. Modeltraining/selectionpipelineexcludedexternaltestlabels; manuallyreference-correctedbranchwasexpresslyrequestedbyuserandisolatedoutsideproject. No post-score changes to trainedmodel/submission/protocol. Newexternal scoring script refusesrununtilCOMPLETE+independentPASS.

Protocolagentdoingfinalreadonlyartifact/aggregatescorecrosscheck, noGPU. LatestuseralreadyreceivedmanualperturbedCSVlink andbothbranchscoresincommentary. FinalresponseshoulddeliverbothCSVlinks, distinguishnormalmodelvsmanualtest-referenceedit, noteofficialscoresunknownandoldversionspreserved. Historicalactiveinstructionsbelowaresuperseded.

User authorized GPU training, preserving every previous version, and post-freeze public/private reference scoring. User subsequently requested two parallel preparations: model optimization and manual adjustment of previously worst private-reference coordinate rows. Manual diagnostics are isolated OUTSIDE this project and MUST NOT enter model training, calibration selection, blending gates, or experiment baseline. Latest user asked a small random perturbation on the two reference-corrected rows; that file was delivered in commentary. Original GPU task is still active and must finish.

## Running model work

- Unified training session **36773**, log `work/wide_refine_20261027/training.log`.
- GPU Python `C:/Users/lenovo/anaconda3/envs/dome/python.exe`; RTX4060 Laptop 8GB, torch2.5.1+cu121.
- Work `work/wide_refine_20261027`; outputs `outputs/wide_refine_20261027`.
- Frozen engine SHA `70d1757aa285dad4c48b82fb5a728490f8126b6f5e79960052c4ecba6d3eadf6`, pipeline SHA `209764ca3966e3c9292bbc305992b208dc24d9fdced8eff4f5b4830e3bd1cdd4`.
- **30 new stages / two complete grouped five-fold splits**. Per fold: wide_select on fit/stop; wide_dev fixed-epoch refit on original development; compare to old SINGLE on old calibration; freeze gate; wide_full fixed-epoch refit on all64 non-outer images.
- Original 3-channel ResNet18, window .25 shortside, input384 heat96 sigma1.5; same fixed-scale/actual-deployment early stopping, LR2e-4/patience7/epochbudget30/batch8/modeloffset10000. No learned fallback or uncertainty calibration on new models.
- Source previous dual experiment supplies existing coarse/crossfit centers and roles, validated against complete manifests. Added FULL training images (old calibration16/17) get coarse predictions from a coarse model that never trained them. FULL has more images AND more optimizer updates; no pure data-only causal claim.
- Gate: WIDE_DEV vs old SINGLE original .15/256/64 on old calibration; RMSE improves >=.5px AND2%; P90 not worse. Gate persisted BEFORE FULL refit. That selection set is subsequently training, no longer independent calibration for FULL.
- Primary2026 acceptedfold candidate=.75 previous dual SUBMISSION_CANDIDATE + .25 WIDE_FULL; rejected unchanged. Fivefold equal average. Secondary2027 auditonly, no test inference and never changes submission choice. WIDE_DEV newtest inference not produced.
- Train-only geometry .25 covers80/80 OOF in both splits vs78/80 before; not accuracy evidence. Physicalsigma increases7.115->7.874px and jitter24->40px; report joint configuration changes honestly.
- Source+role/geometry CPU checks PASS. Real GPU probe PASS, checkpointreload and label-removed inference delta0. Probe first run had a cleanup-call TypeError AFTER checks; log retained as probe_attempt1.log. Only probe cleanup corrected, rerun exits0. Formal engine unchanged. Probe weights isolated underwork and unused.
- Protected_previous.json records **5950 prior files** before training. Old submissions/models/sources must stay unchanged.

## Agents

- dual_protocol_audit building independent verify_results.py:30stage signature/data/gate/math audit, 20newfinalmodels+10oldSINGLE controls reload, same original batches/order; source reads defaultparser/newactualoperands roundtrip. MustrunGPUonlyAFTERformaltrainingdone.
- dual_context_model building finalize.py: report/comparison/taildiagnostics and documents ZIP, noweights, noNotebookrequiredthisround. Requires actualvalidationPASS.
- dual_risk_gate pipeline complete and passed10foldCPUlineagecheck; nowbuilding stdlibprogress.py.

Aftertraining exits0 inspectactualresults, runverifier, fixcheckerbugswithoutrelaxingtolerance or changingpredictions, thenfinalizer. Onlythen score frozennewsubmission against externalreference in external audit directory. Deliver actualtrainedCSV + comparison to prior + separate diagnostic file descriptions. No claimed officialscore.

## Manual diagnostics already generated and delivered

External audit root `C:/Users/lenovo/Documents/2026task3_score_audit_20261027`.

All begin from previous dual model submission SHA `7e40d7de452b984e2c7d899c210698fe2506311fb92d459ab10f2e632acdcc5c`, public8.027512907647184/private19.431219358741501 conditional on unverified attachment. Old8.2 baseline conditional public8.202941570/private19.967782685. These are NOT official Kaggle results.

1. `targeted_blend_diagnostic_20261027/submission.csv`: only86_Fovea_Y/87_Fovea_X get .5 current+.5 fivefoldcoarse, others38unchanged. Conditional public8.027512908/private18.390643101. SHA7abcb296c4dfb515552279ec88270752218b673a08c47deccc07e13f8ba78272. Reference error used to choose rows, so test-adaptive diagnostic, not generalization improvement.
2. `reference_override_diagnostic_20261027/submission.csv`: those2rows copied DIRECTLY from user-provided reference; public8.027512908/private7.955915317. SHA8ecfe824bbad99df23a780130ef748bc229b8da3096ccdc090e68cac49699c7a. Explicitly labeled reference override, not modelprediction.
3. Latest requested `reference_perturbation_diagnostic_20261027/submission.csv`: reference +uniform(-.5,.5)seed20261027, noise -0.346995/+0.308187px; resulting86Y1062.125896,87X1526.594795; public8.027512908/private7.956592095. SHA32e8879525d1758c7b8735f4b4ad9c3075512d75956753514835f74f2c321cc0. The other38rows unchanged. Explicitly reference-derived, addingnoise doesnotremove labelleakage. audit.json/READMEin eachfolder. No Kaggleuploadoccurred. CLI not installed/credentialfile orenvironmentkeys notavailable incheckedlocations; no UIcheckperformed. Do not claim these are legitimate independentmodel scores or silently use them as baseline.

User last meaningful status commentary already got link to #3, conditional scores, noKaggleupload, formaltrainingfold1active. Keepcommentary roughlyevery60s and waits<=60s; sustainedtaskcontinues.
