import warnings
from pathlib import Path

import hydra
import torch
import torchaudio
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.transforms.mel_spectrogram import MelSpectrogramConfig

warnings.filterwarnings("ignore", category=UserWarning)

# load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Synthesize audio from mel-spectrograms using trained vocoder.
    
    Args:
        config: Hydra config with checkpoint, input_dir, output_dir specified via CLI.
    """
    # get paths from config
    checkpoint_path = OmegaConf.select(config, "checkpoint", default=None)
    input_dir = OmegaConf.select(config, "input_dir", default=None)
    output_dir = OmegaConf.select(config, "output_dir", default=None)
    
    if checkpoint_path is None:
        raise ValueError("checkpoint path must be provided")
    if input_dir is None:
        raise ValueError("input_dir must be provided")
    if output_dir is None:
        raise ValueError("output_dir must be ")
    
    checkpoint_path = Path(checkpoint_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # get config from checkpoint or use provided config
    if "config" in checkpoint:
        checkpoint_config = checkpoint["config"]
        model_config = checkpoint_config.get("model", config.model)
    else:
        model_config = config.model
        print("Warning: No config found in checkpoint, using provided config")
    
    # init model
    print("Initializing model...")
    generator = instantiate(model_config).to(device)
    
    # load weights
    if "state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded model weights from epoch {epoch}")
    else:
        generator.load_state_dict(checkpoint)
        print("Loaded model weights (no epoch info)")
    
    generator.eval()
    
    # get mel config from checkpoint or config
    if "config" in checkpoint and "loss_function" in checkpoint["config"]:
        loss_config = checkpoint["config"]["loss_function"]
        sample_rate = loss_config.get("sample_rate", 22050)
        mel_config = MelSpectrogramConfig(
            sr=loss_config.get("sample_rate", 22050),
            win_length=loss_config.get("win_length", 1024),
            hop_length=loss_config.get("hop_length", 256),
            n_fft=loss_config.get("n_fft", 1024),
            f_min=loss_config.get("f_min", 0.0),
            f_max=loss_config.get("f_max", 11025),
            n_mels=loss_config.get("n_mels", 80),
        )
    else:
        # use config (baseline.yaml)
        loss_config = config.loss_function
        sample_rate = OmegaConf.select(loss_config, "sample_rate", default=22050)
        mel_config = MelSpectrogramConfig(
            sr=sample_rate,
            win_length=OmegaConf.select(loss_config, "win_length", default=1024),
            hop_length=OmegaConf.select(loss_config, "hop_length", default=256),
            n_fft=OmegaConf.select(loss_config, "n_fft", default=1024),
            f_min=OmegaConf.select(loss_config, "f_min", default=0.0),
            f_max=OmegaConf.select(loss_config, "f_max", default=11025),
            n_mels=OmegaConf.select(loss_config, "n_mels", default=80),
        )
    
    # create dataset (CustomDirDataset)
    print(f"Loading dataset from: {input_dir}")
    dataset = CustomDirDataset(
        root_dir=str(input_dir),
        target_sr=sample_rate,
        mel_config=mel_config,
        train=True,
        train_ratio=1.0,  
        shuffle_index=False,
    )
    
    print(f"Found {len(dataset)} audio files")
    
    # synth audio
    print(f"Synthesizing audio and saving to: {output_dir}")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Synthesizing"):
            item = dataset[idx]
            mel_spec = item["mel_spec"].to(device)
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)  # [n_mels, T] -> [1, n_mels, T]
            # already [1, n_mels, T]
            audio_path = item.get("audio_path", f"audio_{idx}.wav")
            
            # gen waveform
            waveform_pred = generator(mel_spec)  # [1, L]
            waveform_pred = waveform_pred.squeeze(0).cpu()  # [L]
            
            # clamp [-1, 1]
            waveform_pred = torch.clamp(waveform_pred, -1.0, 1.0)
            
            # orig filename
            original_path = Path(audio_path)
            original_name = original_path.stem  # filename without extension
            
            # save gen audio
            output_path = output_dir / f"{original_name}_synthesized.wav"
            
            # check waveform is 2D
            if waveform_pred.dim() == 1:
                waveform_pred = waveform_pred.unsqueeze(0)  # [1, L]
            
            torchaudio.save(
                str(output_path),
                waveform_pred,
                sample_rate=sample_rate,
            )
    
    print(f"\n✓ Synthesis complete. Generated {len(dataset)} audio files in {output_dir}")
    print(f"  Original files: {input_dir}/audio/")
    print(f"  Synthesized files: {output_dir}/")


if __name__ == "__main__":
    main()
