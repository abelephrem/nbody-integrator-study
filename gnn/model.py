"""GNN Stage B: one-round message-passing model with three message designs."""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

# MLP inputs per design (masses and distance as logs, then standardised):
# 1 = log m_j, disp(3), dist, log dist
# 2 = log m_j, log dist
# 3 = log m_i + log m_j, log(m_i + m_j), log dist
N_IN = {1: 6, 2: 2, 3: 3}
# MLP outputs per design:
# 1 = a free 3-vector
# 2/3 = one strength number s 
N_OUT = {1: 3, 2: 1, 3: 1}


class GravityGNN(MessagePassing):  # inherits PyG's message-passing machinery(gather, message, sum)
    def __init__(self, design: int, hidden: int=128):
        super().__init__(aggr="add")  # set up the parent class; "add" = sum the incoming messages at each receiver
        self.design = design  # remember which design (1, 2 or 3) this model uses
        self.mlp = nn.Sequential( # layers applied one after another, like function composition
            nn.Linear(N_IN[self.design], hidden), nn.SiLU(), # inputs -> 128 numbers, then a smooth nonlinearity
            nn.Linear(hidden, hidden), nn.SiLU(),  # 128 -> 128, second hidden layer
            nn.Linear(hidden, N_OUT[self.design]), # 128 --> output(3 or 1); no nonlinearity so any value is possible
            )
        # per-input shift and scale for standardising; set from the training data by set_input_stats
        self.register_buffer("in_mean", torch.zeros(N_IN[self.design]))  # starts at 0: no shift
        self.register_buffer("in_std", torch.ones(N_IN[self.design]))  # starts at 1: no scaling

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)  # PyG: build per-edge inputs, call message, sum at receivers

    def mlp_inputs(self, x_i, x_j, edge_attr):
        """The MLP's raw inputs for every edge, before standardising: (E, N_IN)."""
        disp = edge_attr[:, :3]  # (E, 3) r_sender - r_receiver
        dist = edge_attr[:, 3:]  # (E, 1) |disp|
        log_dist = dist.log()  # (E, 1)

        if self.design == 1:
            return torch.cat([x_j.log(), disp, dist, log_dist], dim=-1)  # (E, 6)
        if self.design == 2:
            return torch.cat([x_j.log(), log_dist], dim=-1)  # (E, 2)
        return torch.cat([x_i.log() + x_j.log(), (x_i + x_j).log(), log_dist], dim=-1)  # (E, 3) all unchanged if i and j swap

    def set_input_stats(self, graphs) -> None:
        """Measure each input's mean and spread on the given (training) graohs and store them in the buffers."""
        with torch.no_grad():  # just measuring: no gradients needed
            inp = torch.cat([self.mlp_inputs(g.x[g.edge_index[1]], g.x[g.edge_index[0]], g.edge_attr)
                             for g in graphs])  # (all edges, N_IN) raw inputs of every edge in graph

        mean = inp.mean(dim=0) # (N_IN,) average of each column
        std = inp.std(dim=0)  #  (N_IN,) spread of each column

        if self.design == 1:  # disp = columns 1, 2, 3: no shift, one shared scale (decision 12, option C)
            mean[1:4] = 0.0  # keep 0 meaning 0, so swapping the bodies still exactly flips the sign
            std[1:4] = inp[:, 1:4].pow(2).mean().sqrt()  # one length scale for x, y, z together, so directions are kept

        self.in_mean.copy_(mean)  # overwrite the buffers in place
        self.in_std.copy_(std)
            
    def message(self, x_i, x_j, edge_attr):
        inp = (self.mlp_inputs(x_i, x_j, edge_attr) - self.in_mean) / self.in_std  # standardise every column
        out = self.mlp(inp)  # design 1: (E, 3) free vector; designs 2 and 3: (E, 1) strength s

        if self.design == 1:
            return out  # nothing built in

        d_hat = edge_attr[:, :3] / edge_attr[:, 3:]  # (E, 3) unit vector towards the sender (attractive)

        if self.design == 2:
            return out.exp() * d_hat  # s = log of the receiver's acceleration
        return out.exp() / x_i * d_hat  # s = log of the force; divide by the receiver's mass



if __name__ == "__main__":
    from gnn.graph_data import complete_edge_index  # only needed for this check
    torch.manual_seed(0)  # same random numbers every run, so the output is repeatable
    N = 4  # a small made-up system of 4 bodies
    x = torch.rand(N, 1) + 0.5  # (N, 1) random masses between 0.5 and 1.5
    pos = torch.randn(N, 3)  # (N, 3) random positions
    edge_index = complete_edge_index(N)  # every ordered pair, as in Stage A
    send, recv = edge_index  # row 0 = senders, row 1 = receivers
    disp = pos[send] - pos[recv]  # (E, 3) r_sender - r_receiver, same convention as Stage A
    edge_attr = torch.cat([disp, disp.norm(dim=-1, keepdim=True)], dim=-1)  # (E, 4) same layout as Stage A
    for d in (1, 2, 3):
        a = GravityGNN(d)(x, edge_index, edge_attr)  # build a model and call it in one go
        print(d, a.shape, (x * a).sum(dim=0))  # total of mass x acceleration over bodies = net force on the system


    