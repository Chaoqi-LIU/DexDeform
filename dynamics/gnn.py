import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from pytorch3d.ops import knn_points
from typing import Tuple


NUM_RIGHT_HAND_PARTICLES = 19
RIGHT_HAND_EDGES = np.array([
    [17, 18],[18, 17],[16, 17],[17, 16],[16, 18],[18, 16],
    [ 4,  5],[ 5,  4],[ 3,  4],[ 4,  3],[ 3,  5],[ 5,  3],
    [ 7,  8],[ 8,  7],[ 6,  7],[ 7,  6],[ 6,  8],[ 8,  6],
    [10, 11],[11, 10],[ 9, 10],[10,  9],[ 9, 11],[11,  9],
    [14, 15],[15, 14],[13, 14],[14, 13],[13, 15],[15, 13],
    [16,  3],[ 3, 16],[ 3,  6],[ 6,  3],[ 6,  9],[ 9,  6],
    [ 9, 13],[13,  9],[ 1, 16],[16,  1],[ 1,  3],[ 3,  1],
    [ 1,  6],[ 6,  1],[ 1,  9],[ 9,  1],[ 1, 13],[13,  1],
    [12, 16],[16, 12],[12,  3],[ 3, 12],[12,  6],[ 6, 12],
    [12,  9],[ 9, 12],[12, 13],[13, 12],[ 1, 12],[12,  1],
    [ 1,  2],[ 2,  1],[ 2, 12],[12,  2],[ 0,  1],[ 1,  0],
    [ 0,  2],[ 2,  0],[ 0, 12],[12,  0],[ 0, 16],[16,  0]])


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )


    def forward(self, x: torch.Tensor):
        return self.model(x)



class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        x = self.fc(x)
        if residual is not None:
            x += residual
        return self.relu(x)



class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()
        
        self.input_size = input_size    
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, x: torch.Tensor):
        return self.model(x)



# DEBUG = os.environ.get("DEBUG", False)
DEBUG = False
class GraphNeuralDynamics(MessagePassing):
    def __init__(self,
        num_prop_steps: int = 3,
        use_velocity: bool = False,
        latent_dim: int = 64,
        history_window_size: int = 3,
        connectivity_radius: float = 0.1,
        max_num_neighbors: int = 5,
    ):
        super(GraphNeuralDynamics, self).__init__(aggr="add")

        self.num_prop_steps = num_prop_steps
        self.latent_dim = latent_dim
        self.history_window_size = history_window_size
        self.use_velocity = use_velocity
        self.connectivity_radius = connectivity_radius
        self.max_num_neighbors = max_num_neighbors

        self.node_encoder = Encoder(
            input_size=(
                ((3*(history_window_size-1)) if use_velocity else 0) +      # velocity
                3 +                                                         # action
                1                                                           # particle type
            ),
            hidden_size=self.latent_dim,
            output_size=self.latent_dim
        )

        self.edge_encoder = Encoder(
            input_size=(
                3 * self.history_window_size +                              # relative displacement
                1 * 2                                                       # r&s particle type
            ),
            hidden_size=self.latent_dim,
            output_size=self.latent_dim
        )

        # intput: node encoding + aggregated edge effect
        # output: particle effect
        self.node_propagator = Propagator(
            input_size=2 * self.latent_dim,
            output_size=self.latent_dim
        )

        # input: edge encoding + r/s effect
        # output: edge effect
        self.edge_propagator = Propagator(
            input_size=3 * self.latent_dim,
            output_size=self.latent_dim
        )

        # intput: particle effect
        # output: increment of particle position
        self.node_predictor = ParticlePredictor(
            input_size=self.latent_dim,
            hidden_size=self.latent_dim,
            output_size=3
        )

        
    # override MessagePassing class method
    def message(self, x_i, x_j, edges):
        return self.relation_propagator(torch.cat([
            x_i, x_j, edges
        ], dim=-1))
    
    
    def forward(
        self,
        particles: torch.Tensor,
        action: torch.Tensor,
        particle_types: torch.Tensor,
    ) -> torch.Tensor:
        # particles: (B, N, T, 3), N = N_obj + N_hand
        # action: (B, N, 3)
        # particle_types: (B, N,)
        # return: predicted_particles (B, N, 3)

        # check particles format
        assert (
            particle_types[:, :-NUM_RIGHT_HAND_PARTICLES].sum() == 0 and
            particle_types[:, -NUM_RIGHT_HAND_PARTICLES:].sum() == NUM_RIGHT_HAND_PARTICLES * particles.size(0)
        ), "Particles format: [object | hand]"
        
        nodes, edges, edge_indices = self.build_graph(
            particles=particles,
            action=action,
            particle_types=particle_types,
        )

        # encoding
        particle_encoding = self.node_encoder(nodes)
        relation_encoding = self.edge_encoder(edges)

        # propagate
        particle_effect = particle_encoding
        for _ in range(self.num_prop_steps):

            particle_effect = self.node_propagator(torch.cat([
                particle_encoding,
                self.propagate(                 # msg passing + aggregation (handled by pyG)
                    edge_index=edge_indices,
                    x=particle_effect,
                    edges=relation_encoding
                )
            ], dim=-1), residual=particle_effect)

        # predict particles displacement, then forward Euler
        predicted_particles = self.node_predictor(particle_effect) + particles[:, :, -1].view(-1, 3)
        predicted_particles = predicted_particles.view(*particles[:, :, -1].size())

        # no need to predict hand particles
        predicted_particles[particle_types == 1] = particles[particle_types == 1, :, -1]

        return predicted_particles
    

    # override MessagePassing class method
    # NOTE: args' name must not change, see doc for details
    def message(self, x_i, x_j, edges):
        return self.edge_propagator(torch.cat([
            x_i, x_j, edges
        ], dim=-1))
    

    def build_graph(
        self,
        particles: torch.Tensor,
        action: torch.Tensor,
        particle_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # construct graphs, return nodes and edges and edge_indices
        # particles: (B, N, T, 3)
        # action: (B, N, 3)
        # particle_types: (B, N,)

        # connecticity
        edge_indices = self.compute_connecticity(
            particles=particles[:, :, -1],
            radius=self.connectivity_radius,
            max_num_neighbors=self.max_num_neighbors,
        )
        senders, receivers = edge_indices

        # vis scene graph
        if DEBUG:
            N = particles.size(1)
            this_edge_mask = (senders < N) & (receivers < N)
            this_edge_indices = edge_indices[:, this_edge_mask]
            vis_scene_graph(
                particles=particles[0, :, -1],
                edge_indices=this_edge_indices,
                particle_types=particle_types[0]
            )

        # reshape
        particles = particles.view(-1, self.history_window_size, 3)
        action = action.view(-1, 3)
        particle_types = particle_types.view(-1)

        Np = particles.size(0)
        Ne = senders.size(0)

        # node construction
        nodes = []
        nodes.append(action)
        nodes.append(particle_types[:, None])
        if self.use_velocity:
            nodes.append((particles[:, 1:] - particles[:, :-1]).reshape(Np, -1))
        nodes = torch.cat(nodes, dim=-1)
        
        # edge construction
        edges = []
        edges.append((particles[senders] - particles[receivers]).reshape(Ne, -1))
        edges.append(particle_types[senders, None])
        edges.append(particle_types[receivers, None])
        edges = torch.cat(edges, dim=-1)

        return  nodes, edges, edge_indices
    

    def compute_connecticity(
        self,
        particles: torch.Tensor,
        radius: float,
        max_num_neighbors: int = 5,
    ) -> torch.Tensor:
        # particles: (B, N, 3)
        # radius: float
        # max_num_neighbors: int, cap on the number of neighbors
        B, N, _ = particles.size()

        dists, senders, _ = knn_points(
            p1=particles,
            p2=particles,
            norm=2,
            K=max_num_neighbors + 1,
            return_nn=False,
            return_sorted=False
        )
        senders[(dists > radius) | torch.allclose(dists, torch.tensor(0.))] = -1     # (B, N, K)

        # offset indices
        offset = torch.arange(B, device=particles.device) * N
        senders += offset[:, None, None]

        # convert to edge indices
        receivers = torch.arange(B * N, device=particles.device
            ).repeat_interleave(max_num_neighbors + 1)

        # (B, N,) -> (N1 + N2 + ...,)
        senders = senders.reshape(-1)

        # add hand-crafted dex hand edges
        hand_senders = torch.from_numpy(RIGHT_HAND_EDGES[:, 0]).to(particles.device).repeat(B, 1)
        hand_receivers = torch.from_numpy(RIGHT_HAND_EDGES[:, 1]).to(particles.device).repeat(B, 1)
        offset = (torch.arange(B, device=particles.device) + 1) * N - NUM_RIGHT_HAND_PARTICLES
        hand_senders += offset[:, None]
        hand_receivers += offset[:, None]
        hand_senders = hand_senders.reshape(-1)
        hand_receivers = hand_receivers.reshape(-1)
        senders = torch.cat([senders, hand_senders], dim=0)
        receivers = torch.cat([receivers, hand_receivers], dim=0)

        # remove invalid index and self-loop
        mask = (senders != -1) & (senders != receivers)
        senders = senders[mask]
        receivers = receivers[mask]

        return torch.stack([senders, receivers])
    


def vis_scene_graph(
    particles: torch.Tensor,
    edge_indices: torch.Tensor,
    particle_types: torch.Tensor,
):
    # particles: (N, 3)
    # edge_indices: (2, E)
    # particle_types: (N,)

    import open3d as o3d
    
    particles = particles.cpu().numpy()
    edge_indices = edge_indices.cpu().numpy()
    particle_types = particle_types.cpu().numpy()

    N = particles.shape[0]
    # device = particles.device
    
    # adj_matrix = torch.zeros(N, N, device=device)
    # adj_matrix[edge_indices[0], edge_indices[1]] = 1

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(particles)
    colors = np.zeros((N, 3))
    colors[particle_types == 0] = [0, 1, 0]
    colors[particle_types == 1] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # create lines
    lines = o3d.geometry.LineSet()
    lines.points = pcd.points
    lines.lines = o3d.utility.Vector2iVector(edge_indices.T)
    lines.colors = o3d.utility.Vector3dVector(np.zeros((edge_indices.shape[1], 3)))

    o3d.visualization.draw_geometries([pcd, lines])




if __name__ == '__main__':
    B = 3
    N = 128
    T = 5
    gbnd = GraphNeuralDynamics(history_window_size=T, connectivity_radius=5.0)
    particles = torch.randn(B, N, T, 3)
    action = torch.randn(B, N, 3)
    particle_types = torch.zeros(B, N)
    particle_types[:, -NUM_RIGHT_HAND_PARTICLES:] = 1
    predicted_particles = gbnd(particles, action, particle_types)
    print(predicted_particles.size())
        


