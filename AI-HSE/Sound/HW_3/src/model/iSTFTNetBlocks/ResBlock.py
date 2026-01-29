from torch import nn
from ConvBlock import ConvBlock



class ResBlock(nn.Module):
    """
    Residual Block - upsampling time, extracting features (MRF)
    """

    def __init__(self, ch, upsample_coef=8, kernel=..., dilations=...):
        """
        Args:
            upsample_coef (int): coefficient of upsampling time.
            kernel (int): size of conv kernel.
            dilations (List[int]): list of dilations for sequence of conv blocks.
        """
        super().__init__()

        # MRF
        self.conv_block_1 = ConvBlock(dilation=dilations[0])
        self.conv_block_2 = ConvBlock(dilation=dilations[1])
        self.conv_block_3 = ConvBlock(dilation=dilations[2])
        
        self.mrf_extraction = nn.Conv1d(ch*3, ch, kernel_size=1)

    def forward(self, mel_spec, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        # feature extraction
        f1 = self.conv_block_1(mel_spec)
        f2 = self.conv_block_2(mel_spec)
        f3 = self.conv_block_3(mel_spec)

        # sum of features
        f_sum = f1 + f2 + f3

        # mrf
        mrf_extr = self.mrf_extraction(f_sum)
        mrf = F.leaky_relu(mrf_extr, 0.4)

        return mrf + mel_spec


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
