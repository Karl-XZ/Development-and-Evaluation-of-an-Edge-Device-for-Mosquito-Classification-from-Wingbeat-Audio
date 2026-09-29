# PC–ESP32 Parity and Inference-Speed Notes for v4

## Mel and PCEN frontend parity

The v3 correction for PyTorch-compatible Mel-filter interpolation remains in the ESP32 source. v4 adds the same context-local PCEN calculation used during training. Both platforms reset the PCEN smoother at the first frame of each 1.5-second context and use alpha 0.6, delta 2.0, r 0.25, a 0.8-second time constant and epsilon 1e-6.

The final same-PCM v4 validation covered 18 subwindows forming six exact 3.0-second decisions. PC and ESP32 agreed on all subwindow labels and all final aggregated labels. The mean INT8 feature exact-match fraction was 99.9983%, with a maximum difference of one quantization unit. See `validation/PCEN_EXACT3S_PARITY.json`.

## Exact 3.0-second decision geometry

Each live decision captures one fixed 3.0-second interval and evaluates three 1.5-second contexts:

```text
0.00–1.50 s
0.75–2.25 s
1.50–3.00 s
```

The three probability vectors are averaged once. The next live decision starts after the current 3.0-second interval, so both effective support and decision hop are 3.0 seconds. Legacy live aggregation with 3.5 seconds of effective support has been removed from the v4 source.

## Inference-speed patch

The precompiled firmware includes the CMSIS-NN speed patch. The recommended wrapper installs and verifies it automatically before every source rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_flash_included.ps1 -Port COM12
```

The final v4 build measured 667.5 ms mean CNN inference time per 1.5-second context. The package wrapper verified all three patched files by SHA-256 before compiling. The one-command retraining workflow calls the same wrapper after inserting the newly exported TFLite model and C header, so the speed patch is applied automatically for both the included model and a newly trained model.

## Final same-PCM result

The final original-audio model and final flashed firmware were tested on six exact 3.0-second decisions (18 shared PCM contexts). Subwindow-class agreement and aggregated-class agreement were both 100%. Quantized PCEN features were 99.9983% exactly equal, and the maximum difference was one INT8 unit. Full machine-readable results are in `validation/PCEN_EXACT3S_PARITY.json`.

The parity run measured approximately 92 ms of PCEN feature calculation and 668 ms of CNN inference per 1.5-second context. Each decision runs three contexts inside a 3.0-second budget. Timing depends on the exact library and board versions listed in `DEPENDENCIES.txt`.
