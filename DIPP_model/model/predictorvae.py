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
    def __init__(self, future_steps, token_dims, num_cost_weights=6):
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
        positive_weights = F.softplus(raw_weights) + 1e-3             # garantizar positividad
        cost_function_weights = positive_weights.expand(B, -1).contiguous()  # (B, num_cost_weights)

        return actions, cost_function_weights

class PredictorVAE(nn.Module):
    def __init__(self, future_steps, embed_dim=256, num_neighbors=10, num_modes=1, predict_positions=False, use_prior = bool):
        super(PredictorVAE, self).__init__()
        self._future_steps  = future_steps
        self._num_neighbors = num_neighbors 
        self.embed_dim      = embed_dim
        self.pedestrian_net = AgentEncoder(embed_dim=self.embed_dim)    # Encoding the history of each agent: ego and neighbors (embed_dim)
        self.agent_agent_net= Agent2AgentVAE(modes=num_modes, tokens_dim=embed_dim) # Inter agent attention module
        self.ego_decoder = EgoDecoderVAE(future_steps=self._future_steps, token_dims=288)
        self.neighbor_decoder = DecoderVAE(future_steps=self._future_steps, token_dims=288, num_neighbors=num_neighbors, predict_positions=predict_positions)
        self.priori_distribution = mlp_gaussian(input_dim=embed_dim, hidden_dim=256, output_dim=64)  # (B, 64) -> (B, 32) for mu and (B, 32) for logvar
        self.posterior_distribution = mlp_gaussian(input_dim=embed_dim*2, hidden_dim=256, output_dim=64)  # (B, 64) -> (B, 32) for mu and (B, 32) for logvar
        self.reparametrization_train = reparametrization()
        self.use_prior = use_prior

    def forward(self, ego, neighbors, ground_truth):
        """
        ego:       (B, T_obs, feat_dim)          
        neighbors: (B, num_neighbors, T_obs, feat_dim)
        ground_truth: (B, num_neighbors + 1, T_future, feat_dim)
        """
        B = ego.shape[0]
        N = self._num_neighbors

        ego_embed_history = self.pedestrian_net(ego)                 # (B,256)
        ego_embed_future  = self.pedestrian_net(ground_truth[:, 0])  # (B,256)

        # neighbors history: (B,N,T,F) -> (B*N,T,F) -> (B,N,256)
        Bh, Th, Fh = neighbors.shape[0], neighbors.shape[2], neighbors.shape[3]
        nei_hist = neighbors.reshape(B*N, Th, Fh)
        neighbor_embeds_history = self.pedestrian_net(nei_hist).reshape(B, N, -1)  # (B,N,256)

        # neighbors future from GT: ground_truth[:,1:] is (B,N,Tf,F)
        gt_nei = ground_truth[:, 1:1+N]    # (B,N,Tf,F)
        Bf, Tf, Ff = gt_nei.shape[0], gt_nei.shape[2], gt_nei.shape[3]
        nei_fut = gt_nei.reshape(B*N, Tf, Ff)
        neighbor_embeds_future = self.pedestrian_net(nei_fut).reshape(B, N, -1)    # (B,N,256)

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

        """Priori distribution"""
        priori_mu = None
        priori_logvar = None

        if self.use_prior:
            priori = self.priori_distribution(agent_agent_gap)  # (B, 1, 64)
            priori_mu, priori_logvar = priori.chunk(2, dim=-1)  # (B, 1, 32), (B, 1, 32)

        # Posterior distribution
        posterior = self.posterior_distribution(encoder_train)  # (B, 1, 64)
        posterior_mu, posterior_logvar = posterior.chunk(2, dim=-1)
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

        return plans, predictions, cost_function_weights, posterior_mu, posterior_logvar, priori_mu, priori_logvar

    def inference(self, ego, neighbors, z=None, num_samples=1):

        device = ego.device
        B = ego.shape[0]
        N = self._num_neighbors
        latent_dim = 32  

        # =====================================================
        # 1) Encode history
        # =====================================================
        ego_embed_history = self.pedestrian_net(ego)  # (B, 256)

        neighbor_embeds_history = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(N)], dim=1)  # (B, N, 256)

        actors_history = torch.cat([ego_embed_history.unsqueeze(1), neighbor_embeds_history],  dim=1)  # (B, 1+N, 256)

        # =====================================================
        # 2) Build mask exactly as in training
        # =====================================================
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)   # (B,1)
        neighbor_last = neighbors[:, :, -1, :]                          # (B,N,feat_dim)
        neighbor_mask = (neighbor_last[:, :, 6] == 0)                   # (B,N)
        actor_mask = torch.cat([ego_mask, neighbor_mask], dim=1)        # (B,1+N)

        # =====================================================
        # 3) Agent-agent interaction over history only
        # =====================================================
        agent_agent = self.agent_agent_net(actors_history, actor_mask)
        # esperado: (B, 1, 1+N, 256) o compatible con lo que usas luego

        agent_agent_gap = GAP(agent_agent)   # (B, 1, 256)

        # =====================================================
        # 4) Sample z from prior N(0, I)
        # =====================================================
        if z is None:
            z = torch.randn(B, num_samples, latent_dim, device=device)
        else:
            # Aceptamos z con forma (B, latent_dim) y la convertimos a (B,1,latent_dim)
            if z.dim() == 2:
                z = z.unsqueeze(1)

            if z.shape[0] != B:
                raise ValueError(f"z batch mismatch: expected {B}, got {z.shape[0]}")
            if z.shape[-1] != latent_dim:
                raise ValueError(f"z latent_dim mismatch: expected {latent_dim}, got {z.shape[-1]}")

            num_samples = z.shape[1]

        # =====================================================
        # 5) Concatenate z with encoded history for ego
        # =====================================================
        agent_agent_gap_expanded = agent_agent_gap.expand(-1, num_samples, -1)  # (B, num_samples, 256)

        conditionated_encoded = torch.cat(
            [agent_agent_gap_expanded, z],
            dim=-1
        )  # (B, num_samples, 288)

        # =====================================================
        # 6) Decode ego plan for each sample
        # =====================================================
        plans_list = []
        weights_list = []

        for k in range(num_samples):
            cond_k = conditionated_encoded[:, k, :]              # (B, 288)
            plan_k, w_k = self.ego_decoder(cond_k)               # plan_k: (B, T, 2)
            plans_list.append(plan_k.unsqueeze(1))               # (B,1,T,2)
            weights_list.append(w_k.unsqueeze(1))                # (B,1,...) depende de decoder

        plans = torch.cat(plans_list, dim=1)                     # (B, num_samples, T, 2)
        cost_function_weights = torch.cat(weights_list, dim=1)   # (B, num_samples, ...)

        # =====================================================
        # 7) Prepare neighbor tokens
        # =====================================================
        neighbor_tokens = agent_agent[:, :, 1:, :]   # según tu forward actual
        # esperado: (B,1,N,256)

        if neighbor_tokens.shape[1] == 1 and num_samples > 1:
            neighbor_tokens = neighbor_tokens.expand(-1, num_samples, -1, -1)   # (B,num_samples,N,256)

        z_neighbors = z.unsqueeze(2).expand(-1, -1, N, -1)  # (B,num_samples,N,32)

        neighbor_conditioned = torch.cat(
            [neighbor_tokens, z_neighbors],
            dim=-1
        )  # (B,num_samples,N,288)

        current_state_neighbors = neighbors[:, :, -1, :]  # (B,N,feat_dim)

        # =====================================================
        # 8) Decode neighbor futures for each sample
        # =====================================================
        pred_list = []
        for k in range(num_samples):
            neigh_cond_k = neighbor_conditioned[:, k, :, :]                 # (B,N,288)
            pred_k = self.neighbor_decoder(neigh_cond_k, current_state_neighbors)  # (B,N,T,2)
            pred_list.append(pred_k.unsqueeze(1))                           # (B,1,N,T,2)

        predictions = torch.cat(pred_list, dim=1)  # (B,num_samples,N,T,2)

        return plans, predictions, cost_function_weights, z

if __name__ == "__main__":
    model = PredictorVAE(12)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))