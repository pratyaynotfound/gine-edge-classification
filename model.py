import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GINEEdgeModel(nn.Module):

    def __init__(self, node_in_dim=387, hidden_dim=128):
        super().__init__()
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=7
        )
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=7
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        src, dst = edge_index

        h_src = x[src]
        h_dst = x[dst]

        edge_input = torch.cat(
            [h_src, h_dst, edge_attr],
            dim=1
        )

        return self.edge_mlp(edge_input)



