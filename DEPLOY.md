# Deploy the trained classifier for free

Hugging Face now requires Pro for Docker and Gradio Spaces on `cpu-basic`. This project therefore includes a **Static Space** that runs the same CNN locally in the visitor's browser through ONNX Runtime Web. Static Spaces are available to free accounts.

## Free Static Space (recommended)

After `hf auth login`, run:

```sh
python deploy_static_space.py YOUR_USERNAME/audio-genre-classifier --public
```

Replace `YOUR_USERNAME`. The command creates a public Static Space and uploads `static_space/`. Audio never leaves the browser. This version uses the same checkpoint and preprocessing as the Python application; numerical parity checks are recorded in `VALIDATION.md`.

The Docker instructions below are retained for a paid Hugging Face account or another Docker host.

This app needs a Python/PyTorch server. The Dockerfile targets Hugging Face Docker Spaces, whose default application port is 7860. See the [official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker).

## Before deployment

1. Inspect the bundled model’s `models/metrics.json` and `models/confusion_matrix.png`, or substitute the complete artifacts from your own training run.
2. Run `python -m pytest -q`.
3. Start the app with `python -m uvicorn web.app:app --port 7860` and try a real music clip.
4. Confirm `/health` returns HTTP 200. Without trained weights it returns 503, and Docker intentionally refuses to build.

## Hugging Face Spaces

1. Sign in to your Hugging Face account and create a Space named `audio-genre-classifier` with **Docker** as the SDK. Choose the visibility you want and a CPU hardware option; review the displayed cost before selecting hardware.
2. Upload these items to the Space's root, preserving folders:
   - `README.md`, `Dockerfile`, `.dockerignore`, `requirements.txt`, `requirements-cpu.txt`
   - `genre_cnn/` and `web/`
   - `models/`, including the **trained** `cnn.pt` and its metrics
3. Wait for the build to finish. Open the app and classify a valid clip. Check that `/health` is 200 and that invalid/silent uploads produce a helpful error.
4. Add the verified Space URL to the GitHub README and your resume. Do not list a live demo before this check succeeds.

The README already contains the required `sdk: docker` and `app_port: 7860` metadata. Do not upload the GTZAN dataset, a virtual environment, credentials, or spectrogram caches. GitHub's manual upload page does not apply `.gitignore`.

## Optional upload command

If you prefer one upload command, install `huggingface_hub` in your virtual environment and run `hf auth login` on your computer. Then, from the project folder:

```sh
python deploy_space.py YOUR_USERNAME/audio-genre-classifier --public
```

Replace YOUR_USERNAME with your Hugging Face username. This creates a **new public** Docker Space and uploads only the application and trained artifacts. Omit `--public` to keep a new Space private. Existing Spaces are not overwritten. This helper uploads files; you must still wait for a successful build and verify the hosted app. See the [official upload documentation](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Local Docker check

```sh
docker build -t audio-genre-classifier .
docker run --rm -p 7860:7860 audio-genre-classifier
```

The service processes one clip at a time, caps uploads at 40 MB, decodes at most 30 seconds, and does not retain uploaded audio after the request. It supports WAV, FLAC, OGG and MP3 files supported by the installed libsndfile decoder. Only the ten GTZAN genres are recognized; speech, silence, mixed genres and unseen styles are not a reliable use case.
