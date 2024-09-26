from encoder import *
from decoder import *
import config


class FormulaRecognitionModel(nn.Module):
    def __init__(self, growth_rate, hidden_size, attention_dim, vocab_size, max_length=config.max_length):
        super(FormulaRecognitionModel, self).__init__()
        self.encoder = DenseNet(growth_rate)
        self.decoder = AttentionGRUDecoder(hidden_size, attention_dim, vocab_size)
        self.hidden_size = hidden_size
        self.max_length = max_length

    def forward(self, x, target_seq=None):
        device = x.device
        self.encoder.to(device)
        self.decoder.to(device)
        encoder_outputs = self.encoder(x)   # encoder_outputs # 32 512 32 32

        batch_size, channels, height, width = encoder_outputs.size()
        encoder_outputs = encoder_outputs.view(batch_size, channels, height * width)
        encoder_outputs = encoder_outputs.permute(0, 2, 1)  # 32 1024 512

        hidden = torch.zeros(self.decoder.n_layers, batch_size, self.hidden_size).to(device)  # 1 32 512

        output_seq, hidden = self.decoder(encoder_outputs, hidden, target_seq, max_length=self.max_length)
        return output_seq
