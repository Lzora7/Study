from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    """
    Dataset for custom directory with audio files and optional transcriptions.
    
    Expected structure:
    NameOfTheDirectoryWithUtterances
    ├── audio
    │   ├── UtteranceID1.wav (may be flac or mp3)
    │   ├── UtteranceID2.wav
    │   └── ...
    └── transcriptions (optional - ground truth)
        ├── UtteranceID1.txt
        ├── UtteranceID2.txt
        └── ...
    """
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        """
        Args:
            audio_dir (str | Path): path to directory containing audio files
            transcription_dir (str | Path | None): path to directory containing transcription files. If None, transcriptions will be empty strings.
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            raise ValueError(f"Audio directory does not exist: {audio_dir}")
        
        data = []
        audio_files = []
        
        # store audio files
        for path in audio_dir.iterdir():
            if path.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a"]:
                audio_files.append(path)

        # store transcriptions for each audio file
        for audio_path in audio_files:
            # Get audio length
            try:
                t_info = torchaudio.info(str(audio_path))
                audio_len = t_info.num_frames / t_info.sample_rate
            except Exception as e:
                # If can't get audio info, skip file
                print(f"Не удалось получить информацию об аудио {audio_path}: {e}")
                continue
            
            entry = {
                "path": str(audio_path.absolute()),
                "text": "",  # default empty
                "utterance_id": audio_path.stem,  # id
                "audio_len": audio_len,  # required by BaseDataset
            }
            
            # try to load transcription if transcription_dir is provided
            if transcription_dir:
                transcription_dir_path = Path(transcription_dir)
                if transcription_dir_path.exists():
                    transc_path = transcription_dir_path / (audio_path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open(encoding='utf-8') as f:
                            entry["text"] = f.read().strip()
            
            data.append(entry)
        
        super().__init__(data, *args, **kwargs)
