import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    """ Fully Connected Block """

    def __init__(self, in_features, out_features, activation=None, bias=False, dropout=None, spectral_norm=False):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x

class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x