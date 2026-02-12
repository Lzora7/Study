from torch import nn


class ConvBlock(nn.Module):
    """
    Сonvolutional Block
    """

    def __init__(self, ch, kernel_sizes, dilations):
        """
        Args:
            ch (int): input channels.
            kernel_sizes (List[int]): list of kernel sizes to use for each Conv.
            dilations (List[int]): list of dilations for each Conv.
        """
        super().__init__()
        

        self.Conv1 = nn.Conv1d(
            ch, 
            ch, 
            kernel_size=kernel_sizes[0], 
            dilation=dilations[0], 
            padding=(kernel_sizes[0]-1)*dilations[0]//2
        )
        self.Conv2 = nn.Conv1d(
            ch, 
            ch, 
            kernel_size=kernel_sizes[1], 
            dilation=dilations[1],
            padding=(kernel_sizes[1]-1)*dilations[1]//2
        )
        self.Conv3 = nn.Conv1d(
            ch, 
            ch, 
            kernel_size=kernel_sizes[2], 
            dilation=dilations[2],
            padding=(kernel_sizes[2]-1)*dilations[2]//2
        )
        self.LeakyRelu = nn.LeakyReLU(0.1)

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
        out = self.Conv1(mel_spec)
        out = self.LeakyRelu(out)
        out = self.Conv2(out)
        out = self.LeakyRelu(out)
        out = self.Conv3(out)   

        return out

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