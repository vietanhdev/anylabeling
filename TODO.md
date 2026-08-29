# TODO

- [ ] Before each binary release, build from clean, dedicated AnyLabeling
  environments and checksum/launch-test the exact uploaded CPU and GPU
  artifacts on Linux, Windows, and Apple Silicon. Run real accelerator
  inference on each available CUDA, DirectML, CoreML, OpenVINO, or NPU device.
- [ ] Validate real inference in isolated accelerator environments on Linux
  NVIDIA CUDA, Windows CUDA and DirectML, and Apple Silicon CoreML after every
  accelerator-runtime change.
- [ ] Continue triaging GitHub Issues for reproducible application crashes;
  fix them one at a time with a regression test and manual verification before
  merging.
- [ ] Run the NPU provider smoke test on physical Intel Core Ultra, Qualcomm
  Snapdragon X, AMD Ryzen AI, and Huawei Ascend hardware as machines become
  available.
