"""Train on recording-disjoint splits and report recording-level metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from genre_cnn.model import GenreCNN, augment


class SpectrogramDataset(Dataset):
    def __init__(self, cache, manifest, split):
        self.cache = Path(cache)
        self.records = [r for r in manifest["tracks"] if r["split"] == split]
        self.index = [
            (i, j) for i, row in enumerate(self.records) for j in range(row["segments"])
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        track, segment = self.index[index]
        record = self.records[track]
        data = np.load(self.cache / record["cache"], mmap_mode="r", allow_pickle=False)
        return (
            torch.from_numpy(np.array(data[segment], dtype=np.float32)),
            record["label"],
            track,
        )


def evaluate(model, loader, device):
    model.eval()
    totals = {}
    counts = {}
    labels = {}
    with torch.inference_mode():
        for features, target, tracks in loader:
            probabilities = model(features.to(device)).softmax(1).cpu().numpy()
            for score, label, track in zip(
                probabilities, target.tolist(), tracks.tolist()
            ):
                totals[track] = totals.get(track, np.zeros_like(score)) + score
                counts[track] = counts.get(track, 0) + 1
                labels[track] = label
    order = sorted(labels)
    actual = np.array([labels[i] for i in order])
    predicted = np.array([np.argmax(totals[i] / counts[i]) for i in order])
    return actual, predicted


def train(
    cache="data/cnn_cache",
    output="models",
    epochs=35,
    batch_size=32,
    patience=7,
    device="auto",
    threads=2,
):
    if epochs < 1 or batch_size < 1 or patience < 1 or threads < 1:
        raise ValueError("Training parameters must be positive")
    cache, output = Path(cache), Path(output)
    if (output / "cnn.pt").exists():
        raise FileExistsError(
            "A model already exists. Choose a new --output directory to preserve it."
        )
    raw_manifest = (cache / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    seed = manifest["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    sets = {
        s: SpectrogramDataset(cache, manifest, s)
        for s in ["train", "validation", "test"]
    }
    if any(not len(dataset) for dataset in sets.values()):
        raise ValueError("All three splits must contain data")
    groups = {s: {r["sha256"] for r in dataset.records} for s, dataset in sets.items()}
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise ValueError("Recording leakage detected in the split manifest")
    loaders = {
        s: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=s == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        for s, dataset in sets.items()
    }
    model = GenreCNN(len(manifest["classes"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    history = []
    print(
        f"Training on {device}; {sum(p.numel() for p in model.parameters()):,} parameters",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        samples = 0
        for features, target, _ in loaders["train"]:
            features = augment(features).to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item() * len(target)
            samples += len(target)
        actual, predicted = evaluate(model, loaders["validation"], device)
        score = float(
            f1_score(
                actual,
                predicted,
                labels=list(range(len(manifest["classes"]))),
                average="macro",
                zero_division=0,
            )
        )
        accuracy = float(accuracy_score(actual, predicted))
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / samples,
                "validation_accuracy": accuracy,
                "validation_macro_f1": score,
            }
        )
        print(
            f"Epoch {epoch:02d}: loss={loss_total / samples:.4f}, validation accuracy={accuracy:.3f}, macro-F1={score:.3f}",
            flush=True,
        )
        if score > best:
            best = score
            stale = 0
            torch.save(
                {
                    "format_version": 1,
                    "trained": True,
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                    "classes": manifest["classes"],
                    "config": manifest["config"],
                    "epoch": epoch,
                    "validation_macro_f1": score,
                    "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                },
                output / "cnn.pt",
            )
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= patience:
            break
    checkpoint = torch.load(output / "cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    # Evaluate the untouched test partition only after selecting the best epoch.
    actual, predicted = evaluate(model, loaders["test"], device)
    classes = manifest["classes"]
    indices = list(range(len(classes)))
    metrics = {
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "test_macro_f1": float(
            f1_score(
                actual, predicted, labels=indices, average="macro", zero_division=0
            )
        ),
        "unit": "recording (mean segment probabilities)",
        "best_epoch": checkpoint["epoch"],
        "best_validation_macro_f1": checkpoint["validation_macro_f1"],
        "classes": classes,
        "recordings": {s: len(d.records) for s, d in sets.items()},
        "segments": {s: len(d) for s, d in sets.items()},
        "manifest_sha256": checkpoint["manifest_sha256"],
        "classification_report": classification_report(
            actual,
            predicted,
            labels=indices,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=indices
        ).tolist(),
        "limitations": [
            "GTZAN has known label and repetition issues.",
            "Splits are recording-disjoint, not verified artist-disjoint.",
            "Scores are not calibrated probabilities of correctness.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "split_manifest.json").write_bytes(raw_manifest)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=indices,
        display_labels=classes,
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("CNN — held-out recording predictions")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(
        f"Test accuracy: {metrics['test_accuracy']:.3f}; macro-F1: {metrics['test_macro_f1']:.3f}",
        flush=True,
    )
    print(f"Saved {output / 'cnn.pt'} and metrics.json", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/cnn_cache")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
