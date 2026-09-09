"""Publish the trained app to the signed-in user's Hugging Face account.

Install huggingface_hub and run `hf auth login` first. This command creates a
new private Space by default; --public explicitly makes a new Space public.
Existing Spaces are never overwritten by this helper.
"""

import argparse
import json
from pathlib import Path
import tempfile
import shutil

from genre_cnn.predict import Predictor


def stage_app(destination):
    root = Path(__file__).resolve().parent
    destination = Path(destination)
    Predictor(root / "models/cnn.pt")
    required = [
        "README.md",
        "Dockerfile",
        ".dockerignore",
        "requirements.txt",
        "requirements-cpu.txt",
        "models/cnn.pt",
        "models/metrics.json",
        "models/history.json",
        "models/split_manifest.json",
        "models/confusion_matrix.png",
        "models/run_info.json",
        "models/README.md",
    ]
    sources = [root / name for name in required]
    sources += list((root / "genre_cnn").glob("*.py"))
    sources += list((root / "web").glob("*.py"))
    sources += [p for p in (root / "web/static").iterdir() if p.is_file()]
    for source in sources:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    metrics = json.loads((root / "models/metrics.json").read_text())
    # The Space needs a concise model card; the full research README lives on GitHub.
    (destination / "README.md").write_text(
        "---\ntitle: Audio Genre Classifier\nemoji: 🎵\ncolorFrom: green\n"
        "colorTo: gray\nsdk: docker\napp_port: 7860\n---\n\n"
        "# Audio Genre Classifier\n\n"
        "Upload a music clip to explore genre predictions from a PyTorch CNN trained on GTZAN.\n\n"
        f"Held-out recording accuracy: **{metrics['test_accuracy']:.2%}**. "
        f"Macro-F1: **{metrics['test_macro_f1']:.4f}**. "
        "Recording-disjoint evaluation; not verified artist-disjoint. Scores are uncalibrated.\n\n"
        "[Source, notebooks and evaluation details](https://github.com/Tanzin2005/audio-genre-classifier)\n\n"
        "![Held-out confusion matrix](models/confusion_matrix.png)\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Your Hugging Face username/new-space-name")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public Space for sharing on your resume",
    )
    args = parser.parse_args()
    if args.repo_id.count("/") != 1:
        parser.error("Use username/space-name")
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    if args.repo_id.split("/")[0] != identity["name"]:
        parser.error("The Space must be in your signed-in personal account")
    with tempfile.TemporaryDirectory(prefix="audio-genre-deploy-") as temporary:
        stage_app(
            temporary
        )  # Validate/copy the complete artifact before remote writes.
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=not args.public,
            exist_ok=False,
        )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=temporary,
            commit_message="Deploy trained GTZAN CNN audio classifier",
        )
    print(f"Uploaded to https://huggingface.co/spaces/{args.repo_id}")
    print(
        "Wait for the Space build to finish, then verify /health and classify a clip."
    )
    print("Upload success does not mean the hosted build has succeeded yet.")


if __name__ == "__main__":
    main()
