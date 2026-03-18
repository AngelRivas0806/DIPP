import torch
from torch import nn
import torch.nn.functional as F

# Número de modos
NUM_MODES = 20

"""Agent history encoder
# Encodes the historical trajectory of a pedestrian using an LSTM network.
# Mas adelante en la clase Predictor, creamos dos instancias de este encoder, una para los
# vecinos(comparten pesos)y una para el ego"""

class AgentEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(AgentEncoder, self).__init__()
        self.embed_init= 16
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(6,self.embed_init)  
        self.motion    = nn.LSTM(self.embed_init, self.embed_dim, 2, batch_first=True)

    # return the last hidden state as the encoded feature, a tensor of shape (batch_size, embed_dim).
    def forward(self, inputs):
        # Apply ReLU activation to the embedded features before feeding them into the LSTM
        embedded = nn.LeakyReLU()(self.embedding(inputs[:, :, :6]))
        traj, _  = self.motion(embedded)
        # Size of traj: (batch_size, seq_len, embed_dim)
        output   = traj[:, -1]
        return output

"""Multi-modal transformer module is used for agent2map, so we commented that code , we modified agent2agent to use it as well"""
class MultiModalTransformer(nn.Module):
    def __init__(self, modes, tokens_dim, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes     = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(tokens_dim, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(tokens_dim), nn.Linear(tokens_dim, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output



class Agent2Agent(nn.Module):
    def __init__(self, modes: int, tokens_dim: int):
        super(Agent2Agent, self).__init__()
        self.modes = modes
        
        # Encoder to process interactions between agents
        encoder_layer = nn.TransformerEncoderLayer(d_model=tokens_dim, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Multi-modal attention to generate multiple interpretations
        # self.multimodal = MultiModalTransformer(modes=modes, output_dim=512)
        
    def forward(self, inputs, mask=None):
        # Step 1: Encode basic interactions
        encoded = self.interaction_net(inputs, src_key_padding_mask=mask)
        # encoded: (batch, num_agents, tokens_dim)

        # Step 2: Generate NUM_MODES modes using self-attention
        # query = encoded  # (batch, num_agents, tokens_dim)
        # output = self.multimodal(query, encoded, encoded, mask)
        # output: (batch, NUM_MODES, num_agents, tokens_dim)
        output = encoded.unsqueeze(1).expand(-1, self.modes, -1, -1)  # (batch, NUM_MODES, num_agents, tokens_dim)
        return output
    

"""Decoders"""

class AgentDecoder(nn.Module):
    def __init__(self, future_steps, token_dims, num_neighbors=10, num_modes=NUM_MODES, predict_positions=False):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        self._num_neighbors = num_neighbors
        self._num_modes = num_modes
        self._predict_positions = predict_positions
        # Output only 2 values (velocities vx, vy) per step
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(token_dims, 128), nn.ELU(),nn.Linear(128, 128), nn.ELU(), nn.Linear(128, future_steps*2))

    def transform(self, prediction, current_state, dt: float = 0.4):
        """
        prediction:    (B, M, N, T, 2)
        current_state: (B, N, feat)     — [x, y, ...]

        predict_positions=False → predicción = (vx, vy) velocidades → dx=vx*dt → integra con cumsum
        predict_positions=True  → predicción = (x, y) posiciones absolutas → sin integración
        """
        if self._predict_positions:
            # La red predice posiciones absolutas directamente
            return prediction  # (B, M, N, T, 2)

        # La red predice (vx, vy) velocidades → integrar desde posición actual
        x0 = current_state[:, :, 0]  # (B, N)
        y0 = current_state[:, :, 1]  # (B, N)

        vx = prediction[:, :, :, :, 0]  # (B, M, N, T)
        vy = prediction[:, :, :, :, 1]

        new_x = x0[:, None, :, None] + torch.cumsum(vx * dt, dim=-1)  # (B, M, N, T)
        new_y = y0[:, None, :, None] + torch.cumsum(vy * dt, dim=-1)

        return torch.stack([new_x, new_y], dim=-1)  # (B, M, N, T, 2)

    def forward(self, agent_agent, current_state):
        B = agent_agent.shape[0]
        # decoded: (B, M, N, T, 2)
        decoded = self.decode(agent_agent).view(B, self._num_modes, self._num_neighbors, self._future_steps, 2)
        trajs = self.transform(decoded, current_state)  # (B, M, N, T, 2)
        return trajs


class PlanDecoder(nn.Module):
    def __init__(self, future_steps, token_dims, num_modes=NUM_MODES, num_cost_weights=5):
        super(PlanDecoder, self).__init__()
        self._future_steps = future_steps
        self._num_modes    = num_modes
        self.interm_dim    = 256
        self._num_cost_weights = num_cost_weights

        # Output control variables (acceleration and direction change) instead of trajectories
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(token_dims, self.interm_dim), nn.ELU(), nn.Linear(self.interm_dim, future_steps*2))

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

    def forward(self, agent_agent):
        B = agent_agent.shape[0]

        # Generate control variables for num_modes modes
        actions = self.control(agent_agent).view(-1, self._num_modes, self._future_steps, 2)

        # cost_weights: generados por MLP desde dummy input fijo
        # El MLP aprende qué pesos son óptimos durante el entrenamiento
        raw_weights = self.cost_mlp(self.dummy_input)           # (1, num_cost_weights)
        positive_weights = F.softplus(raw_weights)              # garantizar positividad
        cost_function_weights = positive_weights.expand(B, -1).contiguous()  # (B, num_cost_weights)

        return actions, cost_function_weights

"""Score module"""

class Score(nn.Module):
    def __init__(self, num_modes, tokens_dim):
        super(Score, self).__init__()
        self._num_modes = num_modes
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(tokens_dim, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, agent_agent):
        # agent_agent: (batch, num_modes, num_agents, 256)
        # Pooling over all agents for each mode
        agent_pooled = torch.max(agent_agent, dim=2)[0]  # (batch, num_modes, 256)
        scores = self.decode(agent_pooled).squeeze(-1)  # (batch, num_modes)
        # scores → (batch, num_modes)
        return scores



class Predictor(nn.Module):
    # Constructor
    def __init__(self, future_steps, embed_dim=256, num_neighbors=10, num_modes=NUM_MODES, predict_positions=False):
        super(Predictor, self).__init__()
        self._future_steps  = future_steps
        self._num_neighbors = num_neighbors
        self._num_modes     = num_modes
        self.embed_dim      = embed_dim
        self.pedestrian_net = AgentEncoder(embed_dim=self.embed_dim)    # Encoding the history of each agent: ego and neighbors (embed_dim)
        self.agent_agent_net= Agent2Agent(modes=num_modes, tokens_dim=embed_dim) # Inter agent attention module
        self.plan_net       = PlanDecoder(future_steps=self._future_steps, token_dims=embed_dim, num_modes=num_modes) # Plan decoder
        self.predict        = AgentDecoder(future_steps=self._future_steps, token_dims=embed_dim, num_neighbors=num_neighbors, num_modes=num_modes, predict_positions=predict_positions) # Agent decoder
        self.score          = Score(num_modes=num_modes, tokens_dim=embed_dim) # Score estimation

    def forward(self, ego, neighbors):
        """
        ego:       (B, T_obs, feat_dim)          
        neighbors: (B, num_neighbors, T_obs, feat_dim)
        """
        B = ego.shape[0]

        ego_embed = self.pedestrian_net(ego)  # (B, 256)

        neighbor_embeds = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(self._num_neighbors)], dim=1)  
        # (B, num_neighbors, 256)

        actors = torch.cat([ego_embed.unsqueeze(1), neighbor_embeds], dim=1)

        device = ego.device

        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B,1)
        neighbor_last = neighbors[:, :, -1, :]             # (B, N, feat_dim)
        neighbor_mask = (neighbor_last[:, :, 6] == 0)      # (B, N)  True = ausente (flag==0)
        actor_mask = torch.cat([ego_mask, neighbor_mask], dim=1)  # (B, 1+N)
        agent_agent = self.agent_agent_net(actors, actor_mask)
        ego_modes = agent_agent[:, :, 0, :]  
        plans, cost_function_weights = self.plan_net(ego_modes, None)
        current_state_neighbors = neighbors[:, :, -1, :]

        # agent_agent[:, :, 1:] → (B, num_modes, num_neighbors, 256)
        predictions = self.predict(
            agent_agent[:, :, 1:],      # embeddings por modo de vecinos
            current_state_neighbors     # estado actual de vecinos
        )
        # predictions: (B, num_modes, num_neighbors, future_steps, 3)
        scores = self.score(agent_agent)  # (B, num_modes) = (B, 3)

        return plans, predictions, scores, cost_function_weights


if __name__ == "__main__":
    model = Predictor(12)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
