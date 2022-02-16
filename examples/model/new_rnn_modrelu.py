import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter

# class Recurrent(Module):
#     """ Class that implements optimization restricted to the Stiefel manifold """
#     def __init__(self, input_size, output_size):
#         w_ih = Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))
#         w_hh = Parameter(torch.empty((gate_size, real_hidden_size), **factory_kwargs))
#         b_ih = Parameter(torch.empty(gate_size, **factory_kwargs))
#         # Second bias vector included for CuDNN compatibility. Only one
#         # bias vector is needed in standard definition.
#         b_hh = Parameter(torch.empty(gate_size, **factory_kwargs))


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out