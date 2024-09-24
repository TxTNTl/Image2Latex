from encoder import *
from decoder import *
from config import *


class FormulaRecognitionModel(nn.Module):
    def __init__(self, max_length=100):
        super(FormulaRecognitionModel, self).__init__()
        self.encoder = DenseNet(basic_dict['growth_rate'])
        self.decoder = AttentionGRUDecoder(basic_dict['hidden_size'], basic_dict['attention_dim'],
                                           basic_dict['vocab_size'])
        self.hidden_size = basic_dict['hidden_size']
        self.max_length = max_length

    def forward(self, x, target_seq=None):
        encoder_outputs = self.encoder(x) # encoder_outputs # 32 512 32 32

        batch_size, channels, height, width = encoder_outputs.size()
        encoder_outputs = encoder_outputs.view(batch_size, channels, height * width)
        encoder_outputs = encoder_outputs.permute(0, 2, 1) # 32 1024 512

        hidden = torch.zeros(self.decoder.n_layers, batch_size, self.hidden_size).to(x.device)  # 1 32 512

        output_seq, hidden = self.decoder(encoder_outputs, hidden, target_seq, max_length=self.max_length)
        return output_seq
