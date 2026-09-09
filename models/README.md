# Trained CNN artifacts

This folder contains the trained GTZAN CNN used by the upload app. The run achieved **81.73% recording-level test accuracy** and **0.8156 macro-F1** on 197 test recordings. The selected checkpoint comes from epoch 35 of 35 completed epochs.

- `cnn.pt`: weights, preprocessing settings, class order, validation score and split hash.
- `metrics.json`: held-out results, classification report and confusion matrix.
- `history.json`: training and validation progress.
- `split_manifest.json`: exact recording membership and exclusions.
- `run_info.json`: environment, data source and artifact hashes.
- `confusion_matrix.png`: held-out recording predictions.

Keep these artifacts together. A reproduction run should use a new `--output` directory to preserve this model. No raw GTZAN audio is shipped. Load only trusted checkpoints; the loader uses PyTorch's restricted `weights_only=True` format.

Limitations: the ten labels do not cover all music. Scores are uncalibrated. Splits are recording-disjoint but not verified artist-disjoint. GTZAN has label problems and near-duplicates that exact duplicate removal does not resolve. See the main README for the evaluation protocol.
