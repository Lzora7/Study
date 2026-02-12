import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def get_padding(kernel_size, dilation=1):
    """Calculate padding for Conv1d to maintain same length."""
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock(nn.Module):
    """

    """

    def __init__(self, ch, kernel_size=3, dilations=[1, 3, 5]):
        """
        Args:
            ch (int): number of channels.
            kernel_size (int): kernel size for all convolutions (default: 3).
            dilations (List[int]): list of dilations for convs1 (default: [1, 3, 5]).
        """
        super().__init__()

        # 1st set of convolutions with different dilations
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1, 
                dilation=dilations[0],
                padding=get_padding(kernel_size, dilations[0])
            )),
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1,
                dilation=dilations[1],
                padding=get_padding(kernel_size, dilations[1])
            )),
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1,
                dilation=dilations[2],
                padding=get_padding(kernel_size, dilations[2])
            ))
        ])

        # 2nd set of convs with dilation=1
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )),
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )),
            weight_norm(nn.Conv1d(
                ch, ch, kernel_size, stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            ))
        ])

    def forward(self, mel_spec, **batch):
        """
        Model forward method

        Args:
            mel_spec (Tensor): input spectrogram [B, C, T]
        Returns:
            out (Tensor): output spectrogram [B, C, T] (same size)
        """
        x = mel_spec
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x  # res con
        return x


    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths

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
