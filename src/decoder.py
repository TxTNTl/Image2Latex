import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import basic_dict


class AttentionGRUDecoder(nn.Module):
    def __init__(self, hidden_size, attention_dim, output_size, n_layers=1, use_dropout=True):
        # 512 256 113
        super(AttentionGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_dropout = use_dropout

        # GRU layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.2 if use_dropout else 0)
        # 512 to 512

        # Attention layers
        self.attention = nn.Linear(hidden_size + output_size, attention_dim * 2)  # 1208 to 1024
        self.attention_combine = nn.Linear(attention_dim + output_size, hidden_size)          # 1208 to 512

        # Output layer to map GRU output to target vocabulary size
        self.fc_out = nn.Linear(hidden_size, output_size)                       # 512 to 696

    def forward(self, encoder_outputs, hidden, target_seq=None, max_length=basic_dict['max_length']):
        # target_seq 32 100 696
        # encoder_outputs 32 1024 512
        # hidden 1 32 512
        batch_size = encoder_outputs.size(0)        # 32
        output_seq = []

        # Initialize the decoder input (start token embedding)
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(encoder_outputs.device)     # 32 1 696
        encoder_outputs = encoder_outputs.float()
        hidden = hidden.float()
        decoder_input = decoder_input.float()

        # Iterate over each time step in the sequence
        for t in range(max_length):
            # Attention mechanism
            # hidden 一直为1 32 512
            temp = torch.cat((decoder_input.squeeze(1), hidden[-1]), dim=1)     # 32 1208
            temp = temp.float()
            attn_weights = F.softmax(
                self.attention(temp),
                dim=1
            )
            # attn_weights.unsqueeze(1) 32 1 1024
            # encoder_outputs 32 1024 512
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # 32 1 512

            # Combine attention output and GRU hidden state, then pass to GRU
            decoder_input = decoder_input.float()
            attn_applied = attn_applied.float()
            gru_input = self.attention_combine(torch.cat((decoder_input.squeeze(1), attn_applied.squeeze(1)), dim=1)) # 32 1208 to 32 512
            gru_input = F.relu(gru_input) # 32 512

            # Run through GRU
            # gru为512 to 512
            output, hidden = self.gru(gru_input.unsqueeze(1), hidden)

            # Generate output (predicted token probabilities)
            output_token = self.fc_out(output.squeeze(1))   # 32 696
            output_seq.append(output_token)

            # Update decoder input to the next predicted token
            if target_seq is not None:
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                decoder_input = output_token.unsqueeze(1)

        # Stack the output sequence
        output_seq = torch.stack(output_seq, dim=1)
        # [batch_size, max_length=100, output_size=113]
        return output_seq, hidden

# Example usage:
# Assuming encoder_outputs is the output from DenseNet, and hidden is the initial hidden state
# hidden_size = output channels from encoder
# attention_dim = size of the attention layer (typically set manually)
# output_size = number of classes in the output sequence (e.g., vocabulary size for formula recognition)
