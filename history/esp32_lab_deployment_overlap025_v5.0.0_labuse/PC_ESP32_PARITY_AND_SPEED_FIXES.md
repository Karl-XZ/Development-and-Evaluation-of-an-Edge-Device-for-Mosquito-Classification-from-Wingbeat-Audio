# PC–ESP32 Parity and Inference-Speed Notes

## v5 dual-frontend check

The v5 firmware defaults to a matched log-Mel model/frontend pair; command `B` selects the included PCEN pair after the current session has stopped. Both pairs use the v4 Mel-filter correction and strict 3.0-second decision geometry. With the same PCM input, the v5 log-Mel check covered 90 subwindows and 30 aggregated decisions: 90/90 and 30/30 class predictions agreed between PC and ESP32. Quantized features matched exactly in 99.9971% of positions, with a maximum difference of one INT8 unit. See `validation/V005_LOGMEL_EXACT3S_PARITY.json`.

The v5 PCEN check used the same 90/30 geometry. Aggregated predictions agreed in 30/30 decisions; subwindow predictions agreed in 89/90. In the one differing subwindow, all 9,664 quantized input values matched, while two nearly tied raw INT8 output values differed by one unit and reversed the subwindow class. The three-window decision remained the same on both platforms. PCEN feature values matched exactly in 99.9990% of positions, with a maximum difference of one INT8 unit. See `validation/V005_PCEN_EXACT3S_PARITY.json`. Mean CNN inference time was about 668 ms per 1.5-second context for both frontends with the speed patch enabled.

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

## Historical v4 same-PCM result

The final original-audio model and final flashed firmware were tested on six exact 3.0-second decisions (18 shared PCM contexts). Subwindow-class agreement and aggregated-class agreement were both 100%. Quantized PCEN features were 99.9983% exactly equal, and the maximum difference was one INT8 unit. Full machine-readable results are in `validation/PCEN_EXACT3S_PARITY.json`.

The parity run measured approximately 92 ms of PCEN feature calculation and 668 ms of CNN inference per 1.5-second context. Each decision runs three contexts inside a 3.0-second budget. Timing depends on the exact library and board versions listed in `DEPENDENCIES.txt`.
