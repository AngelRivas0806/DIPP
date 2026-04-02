import torch
from torch import nn
import torch.nn.functional as F
from model.predictor import *

# Global Average Pooling
def GAP(x):
    # We also use x.max(dim=1) if we want the maximum value
    return x.mean(dim=2)

# MLP to predict the Gaussian parameters, we need two instances, one for the posteriori distribution and one for the prior
class mlp_gaussian(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_gaussian, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# samples of the posteriori or prior distribution
class reparametrization(nn.Module):
    def __init__(self):
        super(reparametrization, self).__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class DecoderVAE(nn.Module):
    def __init__(self, future_steps, token_dims, num_neighbors=10, predict_positions=False):
        super(DecoderVAE, self).__init__()
        self._future_steps = future_steps
        self._num_neighbors = num_neighbors
        self._predict_positions = predict_positions

        # salida: (vx, vy) por paso para cada vecino
        self.decode = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(token_dims, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, future_steps * 2)
        )

    def transform(self, prediction, current_state, dt: float = 0.4):
        """
        prediction:    (B, N, T, 2)
        current_state: (B, N, feat_dim)  [x, y, ...]
        """
        if self._predict_positions:
            return prediction  # (B, N, T, 2)

        state0 = current_state[:, :, 0:2].unsqueeze(2)  # (B, N, 1, 2)
        states = state0 + torch.cumsum(prediction * dt, dim=2)  # (B, N, T, 2)
        return states

    def forward(self, x, current_state):
        """
        x:            (B, N, token_dims)
        current_state:(B, N, feat_dim)

        return:
            trajs:    (B, N, T, 2)
        """
        B = x.shape[0]

        decoded = self.decode(x).view(B, self._num_neighbors, self._future_steps, 2)
        trajs = self.transform(decoded, current_state)
        return trajs
    
class EgoDecoderVAE(nn.Module):
    def __init__(self, future_steps, token_dims, num_cost_weights=5):
        super(EgoDecoderVAE, self).__init__()
        self._future_steps = future_steps

        # salida: controles [a, steering] por paso
        self.decode = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(token_dims, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, future_steps * 2)
        )
         # MLP que genera los pesos de la función de costo a partir de un dummy input fijo
        # Entrada dummy = tensor de unos de tamaño (1, 1) → se aprende qué pesos usar
        self.cost_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_cost_weights)
        )

        # Dummy input fijo (no es parámetro, solo valor constante de relleno)
        self.register_buffer('dummy_input', torch.ones(1, 1, dtype=torch.float32))

    def forward(self, x):
        """
        x: (B, token_dims)
        return:
            actions: (B, future_steps, 2)
        """
        B = x.shape[0]
        actions = self.decode(x).view(B, self._future_steps, 2)

         # cost_weights: generados por MLP desde dummy input fijo
        # El MLP aprende qué pesos son óptimos durante el entrenamiento
        raw_weights = self.cost_mlp(self.dummy_input)           # (1, num_cost_weights)
        positive_weights = F.softplus(raw_weights)              # garantizar positividad
        cost_function_weights = positive_weights.expand(B, -1).contiguous()  # (B, num_cost_weights)

        return actions, cost_function_weights

class PredictorVAE(nn.Module):
    def __init__(self, future_steps, embed_dim=256, num_neighbors=10, num_modes=1, predict_positions=False):
        super(PredictorVAE, self).__init__()
        self._future_steps  = future_steps
        self._num_neighbors = num_neighbors 
        self.embed_dim      = embed_dim
        self.pedestrian_net = AgentEncoder(embed_dim=self.embed_dim)    # Encoding the history of each agent: ego and neighbors (embed_dim)
        self.agent_agent_net= Agent2Agent(modes=num_modes, tokens_dim=embed_dim) # Inter agent attention module
        self.ego_decoder = EgoDecoderVAE(future_steps=self._future_steps, token_dims=288)
        self.neighbor_decoder = DecoderVAE(future_steps=self._future_steps, token_dims=288, num_neighbors=num_neighbors, predict_positions=predict_positions)
        self.priori_distribution = mlp_gaussian(input_dim=embed_dim*2, hidden_dim=256, output_dim=64)  # (B, 64) -> (B, 32) for mu and (B, 32) for logvar
        self.posterior_distribution = mlp_gaussian(input_dim=embed_dim*2, hidden_dim=256, output_dim=64)  # (B, 64) -> (B, 32) for mu and (B, 32) for logvar
        self.reparametrization_train = reparametrization()
        self.score          = Score(num_modes=1, tokens_dim=embed_dim)  # Scoring module for the modes (B, N) - optional, can be used for mode selection or as an auxiliary loss
    def forward(self, ego, neighbors, ground_truth):
        """
        ego:       (B, T_obs, feat_dim)          
        neighbors: (B, num_neighbors, T_obs, feat_dim)
        ground_truth: (B, num_neighbors + 1, T_future, feat_dim)
        """
        B = ego.shape[0]
        
        ego_embed_history = self.pedestrian_net(ego)  # ----> (B, 256)
        ego_embed_future = self.pedestrian_net(ground_truth[:,0])  # ----> (B, 256)

        neighbor_embeds_history = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(self._num_neighbors)], dim=1) # ---> (B, num_neighbors, 256)
        neighbor_embeds_future = torch.stack([self.pedestrian_net(ground_truth[:, 1+i]) for i in range(self._num_neighbors)], dim=1) # ---> (B, num_neighbors + 1, 256)

        actors_history = torch.cat([ego_embed_history.unsqueeze(1), neighbor_embeds_history], dim=1)
        actors_future = torch.cat([ego_embed_future.unsqueeze(1), neighbor_embeds_future], dim=1)

        device = ego.device

        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B,1)
        neighbor_last = neighbors[:, :, -1, :]             # (B, N, feat_dim)
        neighbor_mask = (neighbor_last[:, :, 6] == 0)      # (B, N)  True = ausente (flag==0)
        actor_mask = torch.cat([ego_mask, neighbor_mask], dim=1)  # (B, 1+N)

        agent_agent = self.agent_agent_net(actors_history, actor_mask)

        future_actor_mask = actor_mask  # (B, 1+N)
        agent_agent_future = self.agent_agent_net(actors_future, future_actor_mask)
        # print(agent_agent.shape)
        # print(agent_agent_future.shape)
        agent_agent_gap = GAP(agent_agent)  # (B, 1, 256)
        agent_agent_future_gap = GAP(agent_agent_future)  # (B, 1, 256)

        encoder_train = torch.cat([agent_agent_gap, agent_agent_future_gap], dim=-1)  # (B, 1, 512)

        # Priori distribution (Constante o no)
        # priori = self.priori_distribution(encoder_train)  # (B, 64)
        # priori_mu, priori_logvar = priori[:, :32], priori[:, 32:64]

        #  Posterior distribution 
        posterior = self.posterior_distribution(encoder_train)  # (B, 64)
        posterior_mu, posterior_logvar = posterior[:, :, :32], posterior[:, :, 32:64]
        # print(f"posterior_mu.shape: {posterior_mu.shape}")
        # print(f"posterior_logvar.shape: {posterior_logvar.shape}")
        # Sample z

        z = self.reparametrization_train(posterior_mu, posterior_logvar)

        # print(f"z.shape: {z.shape}")
        # print(f"agent_agent_gap.shape: {agent_agent_gap.shape}")
        # print(f"agent_agent_future_gap.shape: {agent_agent_future_gap.shape}")

        # Concatenate z with encoder_history
        conditionated_encoded = torch.cat([agent_agent_gap, z], dim=-1)  # (B, 256 + 32)
        # print(f"conditionated_encoded.shape: {conditionated_encoded.shape}")

        # ego controls
        plans, cost_function_weights = self.ego_decoder(conditionated_encoded)
        plans = plans.unsqueeze(1)   # (B, 1, T, 2)

        neighbor_tokens = agent_agent[:, :, 1:, :]   # (B,1,N,256)

        z_neighbors = z.unsqueeze(2).expand(-1, -1, self._num_neighbors, -1)  # (B,1,N,32)

        neighbor_conditioned = torch.cat([neighbor_tokens, z_neighbors], dim=-1)  # (B,1,N,288)                  # (B,N,288)

        current_state_neighbors = neighbors[:, :, -1, :]   # (B,N,feat_dim)

        predictions = self.neighbor_decoder(neighbor_conditioned, current_state_neighbors)  # (B,N,T,2)
        predictions = predictions.unsqueeze(1)  # (B,1,N,T,2)

        scores = self.score(agent_agent)  # (B,N)

        return plans, predictions, scores, cost_function_weights, posterior_mu, posterior_logvar


if __name__ == "__main__":
    model = PredictorVAE(12)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))