import logging
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import torchaudio
from torch.utils.data import Dataset

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)

RUSLAN_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC"
RUSLAN_TAR_GZ_FILE_ID = "1Ye9IqnOvCjDc8NdMQol-bu-gWJGJjbB1"
RUSLAN_METADATA_CSV_FILE_ID = "11TD_ZwIOo-Wo75GYv-OWWOS3ABmwmAdK"


class RUSLANDataset(Dataset):
    """
    RUSLAN dataset for vocoder training.
    Audio only: audio/

    Structure:
    ruslan/
    ├── audio/
    │   ├── 001.wav, 002.wav, ...
    │   └── ...
    """

    def __init__(
        self,
        audio_dir: str,
        text_dir: Optional[str] = None,
        target_sr: int = 22050,
        mel_config: Optional[MelSpectrogramConfig] = None,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        shuffle_index: bool = False,
        train: bool = True,
        train_ratio: float = 0.9,
        seed: int = 42,
        download: bool = False,
        data_dir: Optional[str] = None,
    ):
        """
        Args:
            audio_dir (str): Path to directory with audio files (wav)
            text_dir (str | None): Kept for backward compatibility, but not used.
            target_sr (int): Target sample rate (default 22050)
            mel_config (MelSpectrogramConfig): Configuration for mel-spectrogram extraction.
                If None, uses default config.
            limit (int | None): Limit number of samples
            max_audio_length (float | None): Maximum audio length in seconds
            shuffle_index (bool): Whether to shuffle the dataset
            train (bool): If True, use train split, else test split
            train_ratio (float): Ratio of train/test split (default 0.9)
            seed (int): Random seed for train/test split
            download (bool): If True, automatically download RUSLAN dataset if not found
            data_dir (str | None): Base directory for dataset. If None, uses ROOT_PATH/data/datasets/ruslan
        """
        audio_dir = Path(audio_dir)

        # automatic download if directory missing
        if not audio_dir.exists():
            if download:
                logger.info(f"Audio directory not found: {audio_dir}")
                logger.info("Downloading RUSLAN dataset...")
                audio_dir = self._download_ruslan(data_dir)
            else:
                raise ValueError(
                    f"Audio directory does not exist: {audio_dir}. "
                    "Set download=true in config to download automatically."
                )

        # find all audio files
        audio_files = []
        for ext in [".wav", ".flac", ".mp3", ".m4a"]:
            audio_files.extend(list(audio_dir.glob(f"*{ext}")))
            audio_files.extend(list(audio_dir.glob(f"**/*{ext}")))

        if len(audio_files) == 0:
            if download:
                logger.info(f"No audio files in {audio_dir}. Downloading RUSLAN dataset...")
                audio_dir = self._download_ruslan(data_dir)
                audio_files = []
                for ext in [".wav", ".flac", ".mp3", ".m4a"]:
                    audio_files.extend(list(audio_dir.glob(f"*{ext}")))
                    audio_files.extend(list(audio_dir.glob(f"**/*{ext}")))
            if len(audio_files) == 0:
                raise ValueError(f"No audio files found in {audio_dir}")

        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

        # build index
        index = []
        for audio_path in audio_files:
            if audio_path.name.startswith("._"):
                continue
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
                entry = {"path": str(audio_path), "audio_len": duration}
                index.append(entry)
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
                continue

        if len(index) == 0:
            raise ValueError("No valid audio files found")

        # filter by max_audio_length
        if max_audio_length is not None:
            initial_size = len(index)
            index = [item for item in index if item["audio_len"] <= max_audio_length]
            filtered = initial_size - len(index)
            logger.info(
                f"Filtered {filtered} ({filtered / initial_size:.1%}) records longer than "
                f"{max_audio_length} seconds"
            )

        # sort by audio length for consistent train/test split
        index = sorted(index, key=lambda x: x["audio_len"])

        # split into train/test
        random.seed(seed)
        random.shuffle(index)
        split_idx = int(len(index) * train_ratio)

        if train:
            index = index[:split_idx]
            logger.info(f"Using train split: {len(index)} samples")
        else:
            index = index[split_idx:]
            logger.info(f"Using test split: {len(index)} samples")

        # shuffle if requested (after split)
        if shuffle_index:
            random.seed(seed)
            random.shuffle(index)

        # limit number of samples
        if limit is not None:
            index = index[:limit]
            logger.info(f"Limited to {len(index)} samples")

        self._index = index
        self.target_sr = target_sr

        # setup mel-spectrogram extractor
        if mel_config is None:
            mel_config = MelSpectrogramConfig()
        elif isinstance(mel_config, dict):
            mel_config = MelSpectrogramConfig(**mel_config)
        elif not isinstance(mel_config, MelSpectrogramConfig):
            mel_config = MelSpectrogramConfig(**vars(mel_config))

        self.mel_transform = MelSpectrogram(mel_config)

    @staticmethod
    def _download_ruslan(data_dir: Optional[str] = None) -> Path:
        """
        Download RUSLAN dataset from Google Drive folder.
        Source: https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC

        Structure:
        - data_dir = ROOT_PATH / "data" / "datasets" / "ruslan"
        - Archive is downloaded and extracted into data_dir
        - Audio files are placed in data_dir/audio/
        """
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ruslan"
        else:
            data_dir = Path(data_dir)

        data_dir.mkdir(exist_ok=True, parents=True)
        audio_dir = data_dir / "audio"

        import gdown

        # if audio is already filled, just reuse it
        if audio_dir.exists() and any(audio_dir.rglob("*.wav")):
            for p in data_dir.rglob("*.wav"):
                try:
                    p.relative_to(audio_dir)
                except ValueError:
                    p.unlink()
                    logger.info("Removed stray wav: %s", p)
            logger.info("RUSLAN dataset already exists at %s", audio_dir)
            return audio_dir

        # download and unpack archive
        archive_path = data_dir / "RUSLAN.tar.gz"
        if not archive_path.exists():
            logger.info("Downloading RUSLAN.tar.gz from Google Drive...")
            logger.info("Source: %s", RUSLAN_DRIVE_FOLDER_URL)
            gdown.download(
                id=RUSLAN_TAR_GZ_FILE_ID,
                output=str(archive_path),
                quiet=False,
            )
        else:
            logger.info("Using existing archive: %s", archive_path)

        logger.info("Extracting .wav into %s...", audio_dir)
        audio_dir.mkdir(exist_ok=True, parents=True)
        if archive_path.suffix == ".gz" or archive_path.suffixes == [".tar", ".gz"]:
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                seen_basenames = set()
                wav_count = 0
                for member in tar_ref.getmembers():
                    if not member.isfile() or not member.name.lower().endswith(".wav"):
                        continue
                    base = Path(member.name).name
                    dest_name = base
                    if dest_name in seen_basenames:
                        parent_name = Path(member.name).parent.name
                        stem, ext = Path(member.name).stem, Path(member.name).suffix
                        dest_name = f"{stem}_{parent_name}{ext}"
                    seen_basenames.add(dest_name)
                    dest_path = audio_dir / dest_name
                    with tar_ref.extractfile(member) as f:
                        dest_path.write_bytes(f.read())
                    wav_count += 1
                if wav_count == 0:
                    raise RuntimeError(
                        f"No .wav files in archive {archive_path}. Check archive structure."
                    )
                wav_files_count = wav_count
        else:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                seen_basenames = set()
                wav_count = 0
                for info in zip_ref.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".wav"):
                        continue
                    base = Path(info.filename).name
                    dest_name = base
                    if dest_name in seen_basenames:
                        parent_name = Path(info.filename).parent.name
                        stem, ext = Path(info.filename).stem, Path(info.filename).suffix
                        dest_name = f"{stem}_{parent_name}{ext}"
                    seen_basenames.add(dest_name)
                    dest_path = audio_dir / dest_name
                    dest_path.write_bytes(zip_ref.read(info.filename))
                    wav_count += 1
                if wav_count == 0:
                    raise RuntimeError(
                        f"No .wav files in archive {archive_path}. Check archive structure."
                    )
                wav_files_count = wav_count
        archive_path.unlink()
        logger.info("RUSLAN dataset ready: %s (audio, %d files)", audio_dir, wav_files_count)
        return audio_dir

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        """
        Get audio and mel-spectrogram.

        Returns:
            dict with keys:
                - waveform: [T] audio waveform
                - mel_spec: [n_mels, T'] mel-spectrogram
                - audio_path: path to audio file
        """
        data_dict = self._index[idx]
        audio_path = data_dict["path"]

        # load audio
        audio_tensor, sr = torchaudio.load(audio_path)

        # take first channel if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor[0:1, :]

        # resample if needed
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )

        # convert to mono [1, T] -> [T]
        waveform = audio_tensor.squeeze(0)

        # extract mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # [n_mels, T']

        out = {
            "waveform": waveform,  # [T]
            "mel_spec": mel_spec,  # [n_mels, T']
            "audio_path": audio_path,
        }
        if "text" in data_dict:
            out["text"] = data_dict["text"]
        return out
