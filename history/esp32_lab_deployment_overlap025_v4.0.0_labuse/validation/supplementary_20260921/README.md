# Supplementary validation audit, 2026-09-21

These reports extend the original v4 release checks without changing the
model, firmware, frontend parameters, or decision geometry.

- `SELECTED_FOLD_FP32_INT8.json` compares the selected PyTorch FP32 model and
  its deployed TFLite INT8 conversion on the same 2,934 source-domain
  1.5-second contexts.
- `EXPANDED_PC_ESP32_PARITY_180_DECISIONS.json` compares PC and ESP32
  processing on identical PCM for 540 subwindows and 180 exact-3-second
  aggregate decisions.
- `LAB061_ONE_HOUR_RUNTIME_AUDIT.json` audits a 3,600-second recorded session
  containing all 1,200 expected non-overlapping decisions.
- `CONTROLLED_LOGMEL_PCEN_PAIRED.json` compares five-fold log-Mel and PCEN
  ensembles under matched manifests, splits, seeds, CNN, optimizer, epoch
  budget, context geometry, and ensemble rule. The result is domain-dependent:
  PCEN improves R003 by 6.23 Macro-F1 points but trails log-Mel on the source
  locked test, LAB048, LAB053, LAB061, and the frozen LAB069 negative recording.
- `LAB001_UNLABELED_OUTPUT_AUDIT.json` reports prediction distributions for the
  professor-provided 118.56-second LAB001 recording. It intentionally contains
  no accuracy or F1 because reliable window-level ground truth is unavailable.

The expanded parity report contains individual feature/output records and is
therefore larger than the release smoke-test reports.
