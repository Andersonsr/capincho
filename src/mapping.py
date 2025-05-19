import torch.nn as nn


class Mapper(nn.Module):
    def __init__(self, input_size, token_length, output_n):
        super(Mapper, self).__init__()
        self._tied_weights_keys = None
        self.model = nn.Sequential(
            nn.Linear(input_size, (token_length * output_n) // 2),
            nn.LeakyReLU(),
            nn.Linear((token_length * output_n) // 2, (token_length * output_n)),
            nn.LeakyReLU())

    def forward(self, x):
        return self.model(x)


