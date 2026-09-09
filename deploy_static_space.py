"""Create and upload the free browser-based Hugging Face Static Space."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Your Hugging Face username/new-space-name")
    parser.add_argument("--public", action="store_true", help="Create a public resume demo")
    args = parser.parse_args()
    if args.repo_id.count("/") != 1:
        parser.error("Use username/space-name")
    api = HfApi()
    identity = api.whoami()
    if args.repo_id.split("/")[0] != identity["name"]:
        parser.error("The Space must be in your signed-in personal account")
    app_dir = Path(__file__).resolve().parent / "static_space"
    required = ["README.md", "index.html", "style.css", "app.js", "model.onnx"]
    missing = [name for name in required if not (app_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Static Space files: {missing}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=not args.public,
        exist_ok=False,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=app_dir,
        commit_message="Deploy browser-based CNN audio genre classifier",
    )
    print(f"Deployed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
