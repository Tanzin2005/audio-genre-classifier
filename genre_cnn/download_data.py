"""Download and verify the public GTZAN archive used by this experiment."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
SHA256 = "b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531"


def download(output="data"):
    output = Path(output)
    if (output / "genres").exists():
        raise FileExistsError(
            "The genres folder already exists. Use it or choose another --output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "genres.tar.gz"
    if not archive.exists():
        partial = output / "genres.tar.gz.part"
        print(
            "Downloading GTZAN (about 1.2 GB). Allow about 3 GB free disk space.",
            flush=True,
        )
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            partial.open("wb") as target,
        ):
            total = 0
            last = time.monotonic()
            while chunk := response.read(4 * 1024 * 1024):
                target.write(chunk)
                total += len(chunk)
                if time.monotonic() - last > 15:
                    print(f"{total / 1024**2:.0f} MB downloaded", flush=True)
                    last = time.monotonic()
        partial.replace(archive)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(
            "Archive checksum mismatch. Remove genres.tar.gz and download again."
        )
    with tarfile.open(archive) as source:
        members = [
            entry
            for entry in source.getmembers()
            if not any(
                part.startswith("._") for part in PurePosixPath(entry.name).parts
            )
        ]
        source.extractall(output, members=members, filter="data")
    print(
        f'GTZAN extracted. Next: python -m genre_cnn.prepare --data "{output / "genres"}"',
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data")
    download(parser.parse_args().output)


if __name__ == "__main__":
    main()
