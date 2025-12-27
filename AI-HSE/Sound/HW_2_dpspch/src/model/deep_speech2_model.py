import torch
from torch import nn


class DeepSpeech2Model(nn.Module):
    """
    Deep Speech 2 Model
    
    Architecture:
    1. Convolutional layers
    2. Batch Normalization
    3. Bidirectional RNN (LSTM) layers
    4. Linear output layer for vocabulary prediction
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        conv_channels=[32, 32],
        rnn_hidden=800,
        rnn_layers=5,
        dropout=0.1,
        bidirectional=True,
    ):
        """
        Args:
            n_feats (int): number of input features (frequency channels).
            n_tokens (int): number of tokens in the vocabulary.
            conv_channels (list[int]): list of output channels for each conv layer.
            rnn_hidden (int): hidden size for RNN layers (default: 800).
            rnn_layers (int): number of RNN layers (default: 5).
            dropout (float): dropout probability (default: 0.1).
            bidirectional (bool): whether to use bidirectional RNN (default: True).
        """
        super().__init__()
        self.n_feats = n_feats
        self.n_tokens = n_tokens
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # conv layers

        # [batch, n_feats, time] -> [batch, 1, n_feats, time]
        conv_layers = []
        channels = [1] + conv_channels
        for i in range(len(channels) - 1):
            conv_layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
            ])
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # how many times got smaller
        self.time_reduction_factor = 2 ** len(conv_channels)
        
        # calc reduced frequency dim
        reduced_n_feats = n_feats
        for _ in conv_channels:
            reduced_n_feats = (reduced_n_feats + 1) // 2
        
        # RNN needs [batch, time, features]
        self.conv_output_size = conv_channels[-1] * reduced_n_feats
        
        
        # RNN layers

        self.rnn = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Output projection
        rnn_output_size = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.output_projection = nn.Linear(rnn_output_size, n_tokens)
        
        # init weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # LSTM weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
        
        # output projection
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.5)
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram of shape [batch, n_feats, time].
            spectrogram_length (Tensor): original spectrogram lengths [batch].
        Returns:
            output (dict): output dict containing log_probs and transformed lengths.
        """
        batch_size = spectrogram.size(0)
        
        # [batch, n_feats, time] -> [batch, 1, n_feats, time]
        x = spectrogram.unsqueeze(1)
        x = self.conv_layers(x)
        
        # [batch, channels, reduced_n_feats, reduced_time] -> [batch, reduced_time, channels * reduced_n_feats]
        batch_size, channels, reduced_n_feats, reduced_time = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch, reduced_time, channels, reduced_n_feats]
        x = x.contiguous().view(batch_size, reduced_time, channels * reduced_n_feats)
        
        # get sequence ready for RNN
        reduced_lengths = self.transform_input_lengths(spectrogram_length)
        reduced_lengths = torch.clamp(reduced_lengths, min=1)
        reduced_lengths = reduced_lengths.to(x.device)
        
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # RNN forward
        rnn_out, _ = self.rnn(x_packed)
        
        # unpack sequence
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_out, batch_first=True
        )
        
        # [batch, reduced_time, rnn_hidden * 2] -> [batch, reduced_time, n_tokens]
        output = self.output_projection(rnn_out)
        
        log_probs = nn.functional.log_softmax(output, dim=-1)
        
        return {"log_probs": log_probs, "log_probs_length": reduced_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        Transform input lengths after convolution (reduced by time_reduction_factor).

        Args:
            input_lengths (Tensor): original input lengths [batch]
        Returns:
            output_lengths (Tensor): transformed lengths [batch]
        """

        return (input_lengths.float() / self.time_reduction_factor).long()

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

