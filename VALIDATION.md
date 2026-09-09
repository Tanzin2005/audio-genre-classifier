# Validation record

Checked on Python 3.12.14 with CPU PyTorch 2.14.0+cpu, 2026-09-09.

- `python -m pytest -q`: **10 passed**, one upstream AnyIO deprecation warning.
- Python modules and all notebook 04 code cells: syntax checked.
- `node --check web/static/app.js`: passed.
- Tests exercise optimization, checkpoint export, inference, malformed/silent/short audio, upload limits, and missing-model readiness. Synthetic test artifacts are temporary and are not bundled.
- Full GTZAN run: 35 completed epochs; selected epoch 35 using validation macro-F1. Held-out accuracy **81.73%**, macro-F1 **0.8156**, across 197 recordings.
- Dataset archive checksum verified. Preparation excluded 1 corrupt recording and 14 exact decoded-audio duplicates; 985 recordings remained.

The selected trained checkpoint passed the upload API check with a real GTZAN clip (HTTP 200, ten normalized genre scores). The deployment staging helper was run locally and verified to include the matching checkpoint and all runtime files while excluding raw audio and caches. No remote upload was performed.

No live deployment is claimed. Browser visual testing and a Docker build have not been run; Docker is unavailable in this execution environment. Deployment still requires uploading to an authenticated hosting account, a successful hosted build, and a live prediction check.

## Free browser deployment checks

- The PyTorch checkpoint was exported to ONNX with dynamic segment batching.
- ONNX Runtime logits matched PyTorch with maximum absolute error `1.67e-6` on three input segments.
- The JavaScript log-mel implementation matched librosa on a deterministic audio signal: shape `64 × 130`, maximum absolute error `1.20e-5`, mean absolute error `2.45e-7`.
- `node --check static_space/app.js` passed.

The hosted browser application still needs a live audio check after upload.
