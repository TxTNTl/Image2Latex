import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGRUDecoder(nn.Module):
    def __init__(self, hidden_size, attention_dim, output_size, n_layers=1, use_dropout=True):
        super(AttentionGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_dropout = use_dropout

        # GRU layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.2 if use_dropout else 0)

        # Attention layers
        self.attention = nn.Linear(hidden_size + attention_dim, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, hidden_size)

        # Output layer to map GRU output to target vocabulary size
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, hidden, target_seq=None, max_length=100):
        batch_size = encoder_outputs.size(0)
        output_seq = []

        # Initialize the decoder input (start token embedding)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_size).to(encoder_outputs.device)

        # Iterate over each time step in the sequence
        for t in range(max_length):
            # Attention mechanism
            attn_weights = F.softmax(
                self.attention(torch.cat((decoder_input.squeeze(1), hidden[-1]), dim=1)),
                dim=1
            )
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

            # Combine attention output and GRU hidden state, then pass to GRU
            gru_input = self.attention_combine(torch.cat((decoder_input.squeeze(1), attn_applied.squeeze(1)), dim=1))
            gru_input = F.relu(gru_input).unsqueeze(1)

            # Run through GRU
            output, hidden = self.gru(gru_input, hidden)

            # Generate output (predicted token probabilities)
            output_token = self.fc_out(output.squeeze(1))
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
