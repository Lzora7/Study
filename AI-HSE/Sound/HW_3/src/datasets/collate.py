import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    For vocoder datasets (RUSLAN), pads waveforms and mel-spectrograms
    to the maximum length in the batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if "waveform" in dataset_items[0] and "mel_spec" in dataset_items[0]:
        return _collate_vocoder(dataset_items)


def _collate_vocoder(dataset_items: list[dict]):
    """
    Collate function for vocoder datasets (RUSLAN).
    Pads waveforms and mel-spectrograms to max length in batch.
    """
    batch_size = len(dataset_items)
    
    # get waveforms and mel-spectrograms
    waveforms = [item["waveform"] for item in dataset_items]  # each is [T]
    mel_specs = [item["mel_spec"] for item in dataset_items]  # each is [n_mels, T']
    
    # find max lengths
    max_waveform_len = max(w.shape[0] for w in waveforms)
    max_mel_len = max(m.shape[-1] for m in mel_specs)
    
    # Pad waveforms: each [T] -> [B, T]
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_waveform_len - w.shape[0]
        if pad_len > 0:
            w = torch.nn.functional.pad(w, (0, pad_len), mode="constant", value=0)
        padded_waveforms.append(w)
    waveforms_batch = torch.stack(padded_waveforms)  # [B, T]
    
    # pad mel-spectrograms (last dim)
    padded_mel_specs = []
    for m in mel_specs:
        pad_len = max_mel_len - m.shape[-1]
        if pad_len > 0:
            pad_value = -11.5129251
            m = torch.nn.functional.pad(m, (0, pad_len), mode="constant", value=pad_value)
        padded_mel_specs.append(m)
    mel_specs_batch = torch.stack(padded_mel_specs)  # [B, n_mels, T'] or [B, 1, n_mels, T']
    if mel_specs_batch.dim() == 4:
        mel_specs_batch = mel_specs_batch.squeeze(1)  # [B, n_mels, T']
    
    result_batch = {
        "waveform": waveforms_batch,  # [B, T]
        "mel_spec": mel_specs_batch,  # [B, n_mels, T']
        "audio": waveforms_batch,     
    }
    return result_batch
