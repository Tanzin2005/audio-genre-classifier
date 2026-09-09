import argparse
import json
from pathlib import Path
import torch
from genre_cnn.audio import AudioConfig, load_audio, spectrograms
from genre_cnn.model import GenreCNN


class Predictor:
    def __init__(self, checkpoint="models/cnn.pt"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "No trained CNN found. Run the training notebook to create models/cnn.pt."
            )
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1 or data.get("trained") is not True:
            raise ValueError("This file is not a supported trained CNN checkpoint")
        self.classes = data["classes"]
        self.config = AudioConfig(**data["config"])
        self.model = GenreCNN(len(self.classes))
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()
        self.best_epoch = data["epoch"]

    def predict(self, source):
        y = load_audio(source, self.config)
        features = spectrograms(y, self.config)
        with torch.inference_mode():
            scores = self.model(torch.from_numpy(features)).softmax(1).mean(0).numpy()
        order = scores.argsort()[::-1]
        return {
            "genre": self.classes[int(order[0])],
            "scores": [
                {"genre": self.classes[int(i)], "score": float(scores[i])}
                for i in order
            ],
            "seconds_analyzed": round(len(y) / self.config.sample_rate, 2),
            "segments": len(features),
            "note": "Model scores are not calibrated confidence. Only the first 30 seconds are analyzed.",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--checkpoint", default="models/cnn.pt")
    args = parser.parse_args()
    print(json.dumps(Predictor(args.checkpoint).predict(args.audio), indent=2))


if __name__ == "__main__":
    main()
