import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
     # TODO
    # extracted
    spec_list = [item['spectrogram'] for item in dataset_items]
    text_list = [item['text'] for item in dataset_items]
    text_enc_list = [item['text_encoded'] for item in dataset_items]
    audio_paths_list = [item['audio_path'] for item in dataset_items]

    # save actual length
    spec_lengths = torch.tensor([spec.shape[-1] for spec in spec_list])
    text_encoded_lengths = torch.tensor([text.shape[-1] for text in text_enc_list])

    # max length
    spec_length_max = int(spec_lengths.max().item())
    text_encoded_lengths_max = int(text_encoded_lengths.max().item())

    # batch for padded spectrogram
    batch_spectrograms = []
    # handling different formats
    for spec in spec_list:
        if spec.dim() == 3:
            spec = spec.squeeze(0)
        elif spec.dim() != 2:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}, expected 2D [n_feats, time] or 3D [1, n_feats, time]")
        
        n_feats, time_len = spec.shape  # [n_feats, time]
        if time_len < spec_length_max:
            padding = torch.zeros(n_feats, spec_length_max - time_len, device=spec.device, dtype=spec.dtype)
            spec_padded = torch.cat([spec, padding], dim=1)
        else:
            spec_padded = spec
        batch_spectrograms.append(spec_padded)
    # add batch dimension
    batch_spectrograms = torch.stack(batch_spectrograms)

    # batch for padded text_encoded
    batch_text_encoded = []
    for text_enc in text_enc_list:
        text_enc = text_enc.squeeze(0) # [1, text_len] -> [text_len]
        text_len = text_enc.shape[0]
        if text_len < text_encoded_lengths_max:
            padding_text = torch.zeros(text_encoded_lengths_max - text_len, dtype=text_enc.dtype)
            text_enc_padded = torch.cat([text_enc, padding_text], dim=0)
        else:
            text_enc_padded = text_enc
        batch_text_encoded.append(text_enc_padded)
    # add batch dimension
    batch_text_encoded = torch.stack(batch_text_encoded)

    final_batch = {
        "spectrogram": batch_spectrograms,  # [batch, n_feats, max_time]
        "spectrogram_length": spec_lengths,  # [batch]
        "text_encoded": batch_text_encoded,  # [batch, max_text_len]
        "text_encoded_length": text_encoded_lengths,  # [batch]
        "text": text_list,  # list of strings
        "audio_path": audio_paths_list,  # list of strings
    }

    return final_batch 
