import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleRNN(nn.Module):
    def __init__(self, input_dim, output_size, nhidden, encoder_type, device, 
                 append_missing=False, append_timestamps=False):
        super().__init__()
        self.encoder_type = encoder_type
        self.device = device

        self.append_missing = append_missing
        self.append_timestamps = append_timestamps 
        
        self.input_dim = input_dim
        if self.append_missing: 
            self.input_dim = self.input_dim * 2
        if self.append_timestamps:
            self.input_dim = self.input_dim + 1
        
        if self.encoder_type == "GRU":
            self.enc = nn.GRU(self.input_dim, nhidden, batch_first=True)
        if self.encoder_type == "RNN":
            self.enc = nn.RNN(self.input_dim, nhidden, batch_first=True)
        if self.encoder_type == "LSTM":
            self.enc = nn.LSTM(self.input_dim, nhidden, batch_first=True)
    
        self.fc = nn.Linear(nhidden, output_size)
    
    def get_model_dtype(self):
        """Get the dtype of the first parameter in the model."""
        return next(self.enc.parameters()).dtype

    
    def forward(self, data_and_mask, timestamps):
        # data_and_mask shape: [batch_size, seq_len, dim*2]
        # timestamps shape: [batch_size, seq_len]
        data_and_mask = data_and_mask.to(device=self.device, dtype=self.get_model_dtype())
        timestamps = timestamps.to(device=self.device, dtype=self.get_model_dtype())
        
        # Get sequence lengths from timestamps
        # Count non-zero elements for each sequence
        lengths = (timestamps != 0).sum(dim=1).cpu()
        # Handle sequences with all zeros (length should be 1)
        lengths = lengths + 1
                
        if self.append_missing:
            data = data_and_mask
        else:
            # Assuming mask is the second half of the features
            data = data_and_mask[:, :, :data_and_mask.size(2)//2]

        if self.append_timestamps:
            timestamps = torch.unsqueeze(timestamps, dim=-1)
            data = torch.cat([data, timestamps], dim=-1)
        
        # Pack the sequence
        packed_input = pack_padded_sequence(data, lengths, 
                                            batch_first=True, 
                                            enforce_sorted=False)
            
        if self.encoder_type == "LSTM":
            _, (hidden, _) = self.enc(packed_input)
        else: 
            _, hidden = self.enc(packed_input)
        hidden = hidden.squeeze(0)

        # Use the final hidden state
        output = self.fc(hidden.squeeze(0))
        return output