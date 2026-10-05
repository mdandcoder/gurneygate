# Weights

| File | Use | Where |
|---|---|---|
| `intent.npz` | intent MLP (entering / passing / other) on track history; trained on synthetic tracks with `python -c "from gurneygate.intent import train; train()"` | in git |
| `gurneygate-yolo26s.pt` | detector, PyTorch (CPU, CUDA, Apple MPS) | [release v0.1.0](https://github.com/mdandcoder/gurneygate/releases/tag/v0.1.0) |
| `gurneygate-yolo26s_ncnn_model/` | detector, NCNN (Raspberry Pi 5 CPU), FP16 | release v0.1.0, as `gurneygate-yolo26s_ncnn_model.zip` |

```bash
gh release download v0.1.0 --repo mdandcoder/gurneygate --dir weights
unzip weights/gurneygate-yolo26s_ncnn_model.zip -d weights      # only for the NCNN model
```
Detector: YOLO26s, 640 px input, classes `0 transport` (stretcher, wheelchair, bed) and `1 cart`.
Other formats: `yolo export model=weights/gurneygate-yolo26s.pt format=onnx`.

License: AGPL-3.0-only (trained with Ultralytics YOLO).
