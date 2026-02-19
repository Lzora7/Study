#!/usr/bin/env python3
"""
Загрузк предобученного чекпойнта вокодера.
"""

import argparse
from pathlib import Path

import gdown


def convert_google_drive_link(link: str) -> str:
    """Преобразует ссылку вида .../file/d/FILE_ID/view?usp=sharing в .../uc?id=FILE_ID."""
    if "drive.google.com/uc?id=" in link:
        return link
    import re
    m = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", link)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}"
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}"
    return link

DEFAULT_URL = "https://drive.google.com/file/d/1wtN7KYcPbkmn2ovZqdd8NqUrc-4XAfOp/view?usp=sharing"
DEFAULT_OUTPUT = "saved/demo_run/model_best.pth"

def main():
    parser = argparse.ArgumentParser(description="Download pretrained vocoder checkpoint")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Google Drive link",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = convert_google_drive_link(args.url)
    print(f"Downloading from: {url}")
    gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
