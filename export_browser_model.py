"""Re-export the trained PyTorch checkpoint for browser inference."""

from pathlib import Path

import torch

from genre_cnn.model import GenreCNN

ROOT = Path(__file__).resolve().parent
checkpoint = torch.load(ROOT / "models/cnn.pt", map_location="cpu", weights_only=True)
model = GenreCNN(len(checkpoint["classes"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
torch.onnx.export(
    model,
    torch.zeros(1, 1, 64, 130),
    ROOT / "static_space/model.onnx",
    input_names=["spectrograms"],
    output_names=["logits"],
    dynamic_axes={"spectrograms": {0: "segments"}, "logits": {0: "segments"}},
    opset_version=18,
)
print("Saved static_space/model.onnx")
