import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class AttentionGRUDecoder(nn.Module):
    def __init__(self, hidden_size, attention_dim, output_size, n_layers=1, use_dropout=True):
        super(AttentionGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_dropout = use_dropout

        self.attention = nn.Linear(hidden_size + output_size, 64)
        self.attention_combine = nn.Linear(attention_dim + output_size, hidden_size)          # 1208 to 512

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.2 if use_dropout else 0)  # 512 to 512

        self.fc_out = nn.Linear(hidden_size, output_size)                       # 512 to 696

    def forward(self, encoder_outputs, hidden, target_seq=None, max_length=config.max_length):
        # target_seq 32 100 696
        # encoder_outputs 32 1024 512
        # hidden 1 32 512
        batch_size = encoder_outputs.size(0)
        output_seq = torch.zeros(batch_size, max_length, self.output_size, device=encoder_outputs.device)

        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=encoder_outputs.device)

        for t in range(max_length):
            temp = torch.cat((decoder_input.squeeze(1), hidden[-1]), dim=1).float()
            attn_weights = F.softmax(
                self.attention(temp),
                dim=1
            ).to(encoder_outputs.device)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)    # 32 1 512

            decoder_input = decoder_input.float()
            attn_applied = attn_applied.float()
            gru_input = self.attention_combine(torch.cat((decoder_input.squeeze(1), attn_applied.squeeze(1)), dim=1))
            gru_input = F.relu(gru_input)

            output, hidden = self.gru(gru_input.unsqueeze(1), hidden)

            # Generate output (predicted token probabilities)
            output_token = self.fc_out(output.squeeze(1))   # 32 696
            output_seq[:, t, :] = output_token

            # Update decoder input to the next predicted token
            if target_seq is not None:
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                decoder_input = output_token.unsqueeze(1)

        # [batch_size, max_length=100, output_size=113]
        return output_seq, hidden
