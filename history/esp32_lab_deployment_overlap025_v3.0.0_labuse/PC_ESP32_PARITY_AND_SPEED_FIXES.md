# PC–ESP32 Parity and Inference-Speed Fixes in v3.0.0

## 1. PC–ESP32 Frontend and Output Difference

The ESP32 Mel-filter interpolation previously differed from the PyTorch training implementation. In particular, the endpoint behavior of `torch.linspace()` was not reproduced exactly. This changed 129 Mel-filter weights across 43 of the 64 filters and could alter quantized features and predictions.

The ESP32 implementation now reproduces the training Mel weights. Desktop TFLite parity checks use the `BUILTIN_REF` integer backend because desktop XNNPACK can round a small number of int8 outputs one quantization step differently from TensorFlow Lite Micro.

Real-device verification with identical PCM input produced:

- 30 balanced original-domain windows: 100% PC–ESP32 class agreement and 100% accuracy on both devices.
- LAB001: 100% agreement for all 97 subwindow predictions and all 87 three-window aggregate decisions when the PC uses `BUILTIN_REF`.
- Quantized Mel values: 99.9985% exact equality on LAB001; maximum difference of one int8 step.

Detailed machine-readable results are stored in `validation/`.

## 2. Inference-Speed Patch During Source Rebuilds

The precompiled firmware already contains the supplied CMSIS-NN speed patch. When the firmware is rebuilt from source, the patch must first be installed with:

```powershell
powershell -ExecutionPolicy Bypass -File .\speed_patch\install_speed_patch.ps1
```

If this step is missed, the firmware can still compile and run, but CNN inference may take approximately 1.1 seconds. With the verified patch installed, the measured median CNN inference time is approximately 668 ms, with approximately 68 ms for feature extraction.

The model, training configuration, documentation workflow, and acceleration implementation from v2.0.0 remain otherwise unchanged.
