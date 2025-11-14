import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, neurons, add_self_loops=True, normalize=True):
        super().__init__()
        
        self.layers = []
        for k in range(len(neurons)-1):
            self.layers.append(GCNConv(neurons[k], neurons[k+1]))
        self.layers = ListModule(*self.layers)
    
    def forward(self, data, dropout_rates):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        layers_num = len(self.layers)
        for i in range(layers_num-1):
            layer = self.layers[i]
            x = F.relu(layer(x, edge_index, edge_weight))
            x = F.dropout(x, p=dropout_rates[i], training=self.training)
        x = self.layers[layers_num-1](x, edge_index, edge_weight)

        return torch.sigmoid(x)

    
class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    Source: https://github.com/benedekrozemberczki/ClusterGCN/blob/master/src/layers.py
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
