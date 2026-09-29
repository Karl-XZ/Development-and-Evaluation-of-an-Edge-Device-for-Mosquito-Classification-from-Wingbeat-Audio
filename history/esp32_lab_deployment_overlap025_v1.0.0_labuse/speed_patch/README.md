# Three-file inference-speed patch

These three files come from Chirale_TensorFLowLite 2.0.0 and add a per-function GCC `optimize("O3")` attribute to the measured CMSIS-NN hotspots:

- `arm_convolve_s8.c`
- `arm_nn_mat_mult_kernel_s8_s16.c`
- `arm_q7_to_q15_with_offset.c`

Run `install_speed_patch.ps1` from a normal PowerShell session. The installer accepts an optional `-LibraryRoot` argument, checks the expected original SHA-256 hashes, and makes a timestamped backup before copying. It performs no USB, driver, board, or flash operation.
