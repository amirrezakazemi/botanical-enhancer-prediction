import torch
import torch.nn as nn


# Fully Connected Architecture
class FC(nn.Module):
    def __init__(self, input_dim, layers_dim, output_dim, drop_out):
        super(FC, self).__init__()
        self.type_str = 'FC'
        self.FC = None
        self.loss_fn = None
        layers = list()
        all_layers_dim = layers_dim
        all_layers_dim.insert(0, input_dim)
        for i in range(len(all_layers_dim) - 1):
            layers.append(nn.Linear(all_layers_dim[i], all_layers_dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))

        layers.append(nn.Linear(all_layers_dim[-1], output_dim))
        self.FC = nn.Sequential(*layers)

    def forward(self, x):
        y_hat = self.FC(x)
        return y_hat

    def get_loss(self, y, y_hat):
        return self.loss_fn(y_hat, y.unsqueeze(1))

    def reset_weights(self):
        for name, layer in self.FC.named_children():
            for n, l in layer.named_modules():
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()


### Multi Task Architecture
class MT(nn.Module):
    def __init__(self, input_dim, shared_layers_dim, hidden_dim, spec_layers_dim, output_dim, domain_n, drop_out=0.2):
        super(MT, self).__init__()
        self.enc = None
        self.regs = {}
        self.loss_fn = None
        shared_layers = list()
        all_shared_layers_dim = shared_layers_dim
        all_shared_layers_dim.insert(0, input_dim)
        for i in range(len(all_shared_layers_dim) - 1):
            shared_layers.append(nn.Linear(all_shared_layers_dim[i], all_shared_layers_dim[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(p=drop_out))

        shared_layers.append(nn.Linear(all_shared_layers_dim[-1], hidden_dim))
        self.enc = nn.Sequential(*shared_layers)

        all_spec_layers_dim = spec_layers_dim
        all_spec_layers_dim.insert(0, hidden_dim)

        for _ in range(domain_n):
            spec_layers = list()
            for i in range(len(all_spec_layers_dim) - 1):
                spec_layers.append(nn.Linear(all_spec_layers_dim[i], all_spec_layers_dim[i + 1]))
                spec_layers.append(nn.ReLU())
                spec_layers.append(nn.Dropout(p=drop_out))
            spec_layers.append(nn.Linear(all_spec_layers_dim[-1], output_dim))
            self.regs[_] = nn.Sequential(*spec_layers)

    def forward(self, x, d):
        z = self.enc(x)
        y_hat = []
        for i in range(z.shape[0]):
            a = self.regs[d[i].item()](z[i, :])
            y_hat.append(a)
        y_hat = torch.stack(y_hat)
        return y_hat

    def get_loss(self, y, y_hat):
        return self.loss_fn(y_hat, y.unsqueeze(1))

