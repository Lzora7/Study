"""
CustomDirDataset: vocoder dataset with a single root directory.

Structure:
  root_dir/
  ├── audio/           # .wav (or .flac, .mp3, .m4a)
  │   ├── id1.wav
  │   └── ...
  └── transcriptions/  # optional: id1.txt, id2.txt, ...
      (or text/)
"""

import logging
import random
from pathlib import Path
from typing import Optional

import torchaudio
from torch.utils.data import Dataset

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
TRANSCRIPTION_DIR_NAMES = ("transcriptions", "text")


class CustomDirDataset(Dataset):
    """
    Dataset for a custom directory with audio (and optional transcriptions).
    Single root path; looks for audio/ and optionally transcriptions/ or text/ inside.
    Returns the same format as RUSLANDataset: waveform, mel_spec, audio_path, text.
    Mel spectrogram uses the same config as at training time.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 22050,
        mel_config: Optional[MelSpectrogramConfig] = None,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        shuffle_index: bool = False,
        train: bool = True,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Path to root directory (expects audio/ and optionally transcriptions/ or text/ inside).
            target_sr: Target sample rate.
            mel_config: Mel spectrogram config (same as at training). If None, uses default.
            limit: Max number of samples.
            max_audio_length: Max audio length in seconds (longer files are dropped).
            shuffle_index: Whether to shuffle indices after split.
            train: True for train split, False for test split.
            train_ratio: Fraction of data used for train split.
            seed: Random seed for split and shuffle.
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Root directory does not exist: {root}")

        audio_dir = root / "audio"
        if not audio_dir.is_dir():
            raise ValueError(f"Expected directory 'audio' inside root: {audio_dir}")

        # transcriptions: transcriptions/ or text/
        transcription_dir = None
        for name in TRANSCRIPTION_DIR_NAMES:
            cand = root / name
            if cand.is_dir():
                transcription_dir = cand
                break

        # collect all audio files (including in subdirs)
        audio_files = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
            audio_files.extend(audio_dir.glob(f"**/*{ext}"))

        # skip system/metadata files
        audio_files = [p for p in audio_files if not p.name.startswith("._")]

        if not audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")

        logger.info("CustomDir: found %d audio files in %s", len(audio_files), audio_dir)

        index = []
        for audio_path in audio_files:
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
                entry = {"path": str(audio_path), "audio_len": duration}

                text_path = None
                if transcription_dir is not None:
                    text_path = transcription_dir / (audio_path.stem + ".txt")
                if text_path is not None and text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        entry["text"] = f.read().strip()
                else:
                    entry["text"] = ""

                index.append(entry)
            except Exception as e:
                logger.warning("Skip %s: %s", audio_path, e)
                continue

        if not index:
            raise ValueError("No valid audio files in custom dir")

        if max_audio_length is not None:
            before = len(index)
            index = [e for e in index if e["audio_len"] <= max_audio_length]
            logger.info(
                "Filtered %d samples longer than %.1fs",
                before - len(index),
                max_audio_length,
            )

        index = sorted(index, key=lambda x: x["audio_len"])
        random.seed(seed)
        random.shuffle(index)
        split_idx = int(len(index) * train_ratio)
        if train:
            index = index[:split_idx]
            logger.info("CustomDir train: %d samples", len(index))
        else:
            index = index[split_idx:]
            logger.info("CustomDir test: %d samples", len(index))

        if shuffle_index:
            random.seed(seed)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
            logger.info("CustomDir limited to %d samples", len(index))

        self._index = index
        self.target_sr = target_sr

        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        elif isinstance(mel_config, dict):
            mel_config = MelSpectrogramConfig(**mel_config)
        elif not isinstance(mel_config, MelSpectrogramConfig):
            mel_config = MelSpectrogramConfig(**vars(mel_config))

        self.mel_transform = MelSpectrogram(mel_config)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict: waveform [T], mel_spec [n_mels, T'], audio_path, text (if present).
        """
        data_dict = self._index[idx]
        audio_path = data_dict["path"]

        audio_tensor, sr = torchaudio.load(audio_path)
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        waveform = audio_tensor.squeeze(0)
        mel_spec = self.mel_transform(waveform)

        out = {
            "waveform": waveform,
            "mel_spec": mel_spec,
            "audio_path": audio_path,
        }
        if "text" in data_dict:
            out["text"] = data_dict["text"]
        return out
