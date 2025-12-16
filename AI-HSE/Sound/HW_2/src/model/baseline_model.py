from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Conformer
    """

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Full Conformer Encoder
        
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_tokens),
        )

    def _spec_aug_block(self):
        '''
        SpecAug block (first)
        '''

    def _conv_subsampling_block(self):
        '''
        Convolution SubSampling block (second)
        '''

    def _conformer_block(self):
        '''
        Conformer block

        '''

    def __ffn_module(self):
        '''
        FFN module in Conformer Block
        '''
        res_connection = ...

        nn.LayerNorm
        nn.Linear
        nn.SiLU()
        nn.Dropout
        nn.Linear
        nn.Dropout

        out = ... + res_connection

        return out

    def __mhsa_module(self):
        '''
        Multi-Head Self Attention Module in Conformer Block
        '''
        res_connection = ...

        nn.LayerNorm
        nn.MultiheadAttention
        nn.Dropout

        out = ... + res_connection

        return out

    def __conv_module(self):
        '''
        Convolution Module in Conformer Block
        '''   
        res_connection = ...

        nn.LayerNorm
        nn.Conv1d(kernel_size=1) # pointwise conv ?
        nn.GLU()
        nn.Conv1d(kernel_size=1) # 1d depthwise conv
        nn.BatchNorm1d
        nn.SiLU()
        nn.Conv1d(kernel_size=1) # pointwise conv ?
        nn.Dropout

        out = ... + res_connection

        return out
        
    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        output = self.net(spectrogram.transpose(1, 2))
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
