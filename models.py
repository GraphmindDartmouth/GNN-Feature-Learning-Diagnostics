import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GPSConv, SAGEConv, GCN2Conv, AntiSymmetricConv, global_mean_pool, MessagePassing, SGConv
from torch_geometric.utils import softmax, to_dense_batch
from torch_geometric.nn.inits import reset

# MNIST MLP
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 5000)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5000, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

## simple models
## SGC
class SGC(nn.Module):
    def __init__(self, d, h, c, l=3):
        super().__init__()
        self.conv = SGConv(
            in_channels=d,
            out_channels=h,
            K=l,
            cached=False,
        )
        self.final_layer = nn.Linear(h, c)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.conv(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        out = self.final_layer(x) 
        return out

class spMLP(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = nn.Linear(in_features=self.d, out_features=self.h, bias=False)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=self.h, out_features=self.h, bias=False) for _ in range(self.l)]
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index=None, edge_attr=None, batch=None):
        Z = self.input_layer(X)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

class spGCN(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = GCNConv(in_channels=self.d, out_channels=self.h, add_self_loops=False, bias=False)
        self.hidden_layers = nn.ModuleList(
            [GCNConv(in_channels=self.h, out_channels=self.h, add_self_loops=False, bias=False) for _ in range(self.l)]
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index, edge_attr=None, batch=None):
        Z = self.input_layer(X, edge_index)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z, edge_index)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

class spGCN2(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = nn.Linear(self.d, self.h)
        self.hidden_layers = nn.ModuleList(
            GCN2Conv(channels=self.h, alpha=0.2) for _ in range(self.l)
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index, edge_attr=None, batch=None):
        Z = self.input_layer(X)
        Z = self.activation_fn(Z)
        x0 = Z
        for layer in self.hidden_layers:
            Z = layer(Z, x0, edge_index)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

class spSAGE(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = SAGEConv(in_channels=self.d, out_channels=self.h, bias=False)
        self.hidden_layers = nn.ModuleList(
            [SAGEConv(in_channels=self.h, out_channels=self.h, bias=False) for _ in range(self.l)]
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index, edge_attr=None, batch=None):
        Z = self.input_layer(X, edge_index)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z, edge_index)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

class spGAT(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = GATConv(in_channels=self.d, out_channels=self.h, add_self_loops=False, bias=False, heads=4, concat=False)
        self.hidden_layers = nn.ModuleList(
            [GATConv(in_channels=self.h, out_channels=self.h, add_self_loops=False, bias=False, heads=4, concat=False) for _ in range(self.l)]
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index, edge_attr=None, batch=None):
        Z = self.input_layer(X, edge_index)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z, edge_index)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out


class spADGN(nn.Module):
    def __init__(self, d, h, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.input_layer = nn.Linear(self.d, self.h)
        self.hidden_layers = nn.ModuleList(
            AntiSymmetricConv(self.h) for _ in range(self.l)
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, edge_index, edge_attr=None, batch=None):
        Z = self.input_layer(X)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z, edge_index)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

### MLP + PE
class peMLP(nn.Module):
    def __init__(self, d, h, p, c, l):
        super().__init__()
        self.d = d
        self.h = h
        self.p = p
        self.c = c
        self.l = l
        self.activation_fn = nn.ReLU()

        self.pe_lin = nn.Linear(20, p)
        self.pe_norm = nn.BatchNorm1d(20)
        self.input_layer = nn.Linear(in_features=self.d, out_features=self.h-self.p, bias=False)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=self.h, out_features=self.h, bias=False) for _ in range(self.l)]
        )
        self.final_layer = nn.Linear(in_features=self.h, out_features=self.c, bias=False)

    def forward(self, X, pe, edge_index=None, edge_attr=None, batch=None):
        x_pe = self.pe_norm(pe)
        Z = torch.cat((self.input_layer(X), self.pe_lin(x_pe)), 1)
        Z = self.activation_fn(Z)
        for layer in self.hidden_layers:
            Z = layer(Z)
            Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out = global_mean_pool(out, batch)
        return out

## Graph Transformers
## GPS block with MPNN only
class GPSConv2(nn.Module):
    '''
    Wrap MPNN with residual connection and FFN
    '''
    def __init__(
        self,
        channels,
        conv,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
    
    def reset_parameters(self):
        if self.conv is not None:
            self.conv.reset_parameters()
        reset(self.mlp)

    def forward(
        self,
        x,
        edge_index,
        batch = None,
        **kwargs,
    ):
        h = self.conv(x, edge_index, **kwargs)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        out = h + self.mlp(h)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.channels}, conv={self.conv})'
## GPS
class GPS(nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, attn_type: str, num_classes: int, conv_only: bool=False, attn_only: bool=False):
        super().__init__()
        self.node_lin = nn.Linear(3, channels - pe_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mpnn = GATConv(in_channels=channels, out_channels=channels, heads=4, concat=False)
            if attn_only:
                conv = GPSConv(channels, None, heads=4, attn_type=attn_type, norm=None)
            elif conv_only:
                conv = GPSConv2(channels, mpnn)
            else:
                conv = GPSConv(channels, mpnn, heads=4, attn_type=attn_type, norm=None)
            self.convs.append(conv)
        
        self.out = nn.Sequential(
                nn.Linear(channels, channels//4),
                nn.ReLU(),
                nn.Linear(channels//4, channels//8),
                nn.ReLU(),
                nn.Linear(channels//8, num_classes)
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = global_mean_pool(x, batch)
        return self.out(x)

## GCN+
class GCN(nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        self.node_lin = nn.Linear(3, channels - pe_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mpnn = GCNConv(in_channels=channels, out_channels=channels)
            conv = GPSConv2(channels, mpnn)
            self.convs.append(conv)
        
        self.out = nn.Sequential(
                nn.Linear(channels, channels//4),
                nn.ReLU(),
                nn.Linear(channels//4, channels//8),
                nn.ReLU(),
                nn.Linear(channels//8, num_classes)
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = global_mean_pool(x, batch)
        return self.out(x)

## Local GT
class MultiplicativeAttention(MessagePassing):
    def __init__(self, channels, heads=4):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.dim = channels // heads
        self.scale = math.sqrt(self.dim)

        self.lin_q = nn.Linear(channels, channels)
        self.lin_k = nn.Linear(channels, channels)
        self.lin_v = nn.Linear(channels, channels)
        self.lin_out = nn.Linear(channels, channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x, edge_index):
        Q = self.lin_q(x).view(-1, self.heads, self.dim)
        K = self.lin_k(x).view(-1, self.heads, self.dim)
        V = self.lin_v(x).view(-1, self.heads, self.dim)
        h = self.propagate(edge_index, Q=Q, K=K, V=V, size=(x.size(0), x.size(0)))
        h = self.lin_out(h)
        
        h = h + x
        out = h + self.ffn(h)
        return out

    def message(self, Q_i, K_j, V_j, index, ptr, size_i):
        score = (Q_i * K_j).sum(dim=-1) / self.scale
        alpha = softmax(score, index, ptr, size_i)
        return V_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out.view(-1, self.heads * self.dim)

## Local GT
class MultiplicativeAttentionEdge(MessagePassing):
    def __init__(self, channels, heads=4):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.dim = channels // heads
        self.scale = math.sqrt(self.dim)

        self.lin_q = nn.Linear(channels, channels)
        self.lin_k = nn.Linear(channels, channels)
        self.lin_v = nn.Linear(channels, channels)
        self.lin_out = nn.Linear(channels, channels)

        # Edge feature used as gate on V_j
        self.edge_gate = nn.Sequential(
            nn.Linear(channels, heads),
            nn.Sigmoid()
        )

        # Edge feature used to bias attention scores
        self.edge_bias = nn.Sequential(
            nn.Linear(channels, heads),
            nn.Tanh()
        )

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x, edge_index, edge_attr):
        Q = self.lin_q(x).view(-1, self.heads, self.dim)
        K = self.lin_k(x).view(-1, self.heads, self.dim)
        V = self.lin_v(x).view(-1, self.heads, self.dim)

        h = self.propagate(edge_index, Q=Q, K=K, V=V, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        h = self.lin_out(h)

        h = h + x
        out = h + self.ffn(h)
        return out

    def message(self, Q_i, K_j, V_j, edge_attr, index, ptr, size_i):
        score = (Q_i * K_j).sum(dim=-1) / self.scale                     # [E, heads]
        bias = self.edge_bias(edge_attr)                                # [E, heads]
        alpha = softmax(score + bias, index, ptr, size_i)               # [E, heads]

        gate = self.edge_gate(edge_attr).unsqueeze(-1)                  # [E, heads, 1]
        V_j = V_j * gate                                              # gated V_j
        return V_j * alpha.unsqueeze(-1)                              # [E, heads, dim]

    def update(self, aggr_out):
        return aggr_out.view(-1, self.heads * self.dim)

class GraphTransformer(nn.Module):
    def __init__(self, channels, pe_dim, num_classes, num_layers, use_edge):
        super().__init__()
        self.use_edge = use_edge

        self.node_lin = nn.Linear(3, channels - pe_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.edge_lin = nn.Linear(1, channels)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_edge:
                layer = MultiplicativeAttentionEdge(
                    channels=channels,
                    heads=4,
                )
            else:
                layer = MultiplicativeAttention(
                    channels=channels,
                    heads=4,
                )
            self.layers.append(layer)

        self.out = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, num_classes)
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_lin(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_lin(edge_attr.view(-1, 1))
        
        for attn_layer in self.layers:
            if self.use_edge:
                x = attn_layer(x, edge_index, edge_attr)
            else:
                x = attn_layer(x, edge_index)
       
        x = global_mean_pool(x, batch)
        return self.out(x)
    
## TokenGT
class TokenGTLayer(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x, batch):
        h, mask = to_dense_batch(x, batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = h + x
        h = h + self.ffn(h)
        return h

class TokenGT(nn.Module):
    def __init__(self, channels, num_layers, num_classes):
        super().__init__()
        d_feat = int(0.5 * channels)
        d_pe = int(0.15 * channels)
        d_type = channels - (d_feat + 2 * d_pe)

        self.node_proj = nn.Linear(3, d_feat)
        self.edge_embed = nn.Parameter(torch.randn(d_feat))
        self.pe_proj = nn.Linear(20, d_pe)
        self.pe_norm = nn.BatchNorm1d(20)
        self.node_type = nn.Parameter(torch.randn(d_type))
        self.edge_type = nn.Parameter(torch.randn(d_type))

        self.layers = nn.ModuleList([
            TokenGTLayer(channels, heads=4) for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, num_classes)
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        pe = self.pe_norm(pe)
        h_node = torch.cat([
            self.node_proj(x),
            self.pe_proj(pe),
            self.pe_proj(pe),
            self.node_type.expand(x.size(0), -1)
        ], dim=-1)

        src, dst = edge_index
        h_edge = torch.cat([
            self.edge_embed.expand(edge_index.size(1), -1),
            self.pe_proj(pe[src]),
            self.pe_proj(pe[dst]),
            self.edge_type.expand(edge_index.size(1), -1)
        ], dim=-1)

        h_token = torch.cat([h_node, h_edge], dim=0)
        token_batch = torch.cat([batch, batch[src]], dim=0)
        perm = token_batch.argsort()
        h_token = h_token[perm]
        token_batch = token_batch[perm]

        for layer in self.layers:
            h_token = layer(h_token, token_batch)
        out = global_mean_pool(h_token, token_batch)
        return self.out(out)
