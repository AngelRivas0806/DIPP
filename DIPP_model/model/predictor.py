import torch
from torch import nn
import torch.nn.functional as F

# Número de modos
NUM_MODES = 20

"""Agent history encoder
# Encodes the historical trajectory of a pedestrian using an LSTM network."""

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
    # return the last hidden state as the encoded feature, a tensor of shape (batch_size, 256).
    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]

        return output


"""Multi-modal transformer module is used for agen2map, so we commented that code , we modified agent2agent to use it as well"""
class MultiModalTransformer(nn.Module):
    def __init__(self, modes=NUM_MODES, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output



class Agent2Agent(nn.Module):
    def __init__(self, modes=NUM_MODES):
        super(Agent2Agent, self).__init__()
        self.modes = modes
        
        # Encoder to process interactions between agents
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Multi-modal attention to generate multiple interpretations
        self.multimodal = MultiModalTransformer(modes=modes, output_dim=256)
        
    def forward(self, inputs, mask=None):
        # Step 1: Encode basic interactions
        encoded = self.interaction_net(inputs, src_key_padding_mask=mask)
        # encoded: (batch, num_agents, 256)

        # Step 2: Generate NUM_MODES modes using self-attention
        query = encoded  # (batch, num_agents, 256)
        output = self.multimodal(query, encoded, encoded, mask)
        # output: (batch, NUM_MODES, num_agents, 256)
        
        return output
    

"""Decoders"""

class AgentDecoder(nn.Module):
    def __init__(self, future_steps, num_neighbors=10, num_modes=NUM_MODES):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        self._num_neighbors = num_neighbors
        self._num_modes = num_modes
        # Output only 2 values (delta_x, delta_y) per step
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, future_steps*2))

    def transform(self, prediction, current_state):
        x = current_state[:, 0] 
        y = current_state[:, 1]
        # Ignore theta/velocity from input, we only predict position
        
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        
        new_x = x.unsqueeze(1) + delta_x 
        new_y = y.unsqueeze(1) + delta_y 
        
        # Output: [x, y]
        traj = torch.stack([new_x, new_y], dim=-1)

        return traj
       
    def forward(self, agent_agent, current_state):
        # View as [..., future_steps, 2] instead of 3
        decoded = self.decode(agent_agent).view(-1, self._num_modes, self._num_neighbors, self._future_steps, 2)
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(self._num_modes) for j in range(self._num_neighbors)], dim=1)
        trajs = torch.reshape(trajs, (-1, self._num_modes, self._num_neighbors, self._future_steps, 2))

        return trajs



class AVDecoder(nn.Module):
    def __init__(self, future_steps=12, feature_len=9, num_modes=NUM_MODES):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self._num_modes = num_modes
        # Output control variables (acceleration and direction change) instead of trajectories
        self.control = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, future_steps*2)
        )
        
        # All 9 cost weights are now learnable
        self.cost_weights = nn.Parameter(torch.ones(feature_len, dtype=torch.float32))

        self.register_buffer('scale', torch.tensor([
            0.5,   # weight[0]: acceleration_aux 
            0.5,   # weight[1]: acceleration 
            0.1,   # weight[2]: jerk 
            0.1,   # weight[3]: steering 
            0.1,   # weight[4]: steering_change 
            5.0,   # weight[5]: collision_avoidance 
            2.0,   # weight[6]: trajectory_following_aux1
            2.0,   # weight[7]: trajectory_following
            2.0,   # weight[8]: trajectory_following_aux2
        ], dtype=torch.float32))

    def forward(self, agent_agent, current_state):
        # Generate control variables for num_modes modes
        actions = self.control(agent_agent).view(-1, self._num_modes, self._future_steps, 2)

        # Generate cost function weights (all learnable, scaled for pedestrians)
        norm_weights = torch.softmax(self.cost_weights, dim=0)  # (9,)
        scaled_weights = norm_weights * self.scale  # (9,) - all 9 weights are now scaled
        
        # Expand to batch dimension for compatibility
        cost_function_weights = scaled_weights.unsqueeze(0).expand(actions.shape[0], -1)  # (B, 9)


        return actions, cost_function_weights

"""Score module"""

class Score(nn.Module):
    def __init__(self, num_modes=NUM_MODES):
        super(Score, self).__init__()
        self._num_modes = num_modes
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, agent_agent):
        # agent_agent: (batch, num_modes, num_agents, 256)
        # Pooling over all agents for each mode
        agent_pooled = torch.max(agent_agent, dim=2)[0]  # (batch, num_modes, 256)
        scores = self.decode(agent_pooled).squeeze(-1)  # (batch, num_modes)
        # scores → (batch, num_modes)
        return scores



class Predictor(nn.Module):
    def __init__(self, future_steps, num_neighbors=10, num_modes=NUM_MODES):
        super(Predictor, self).__init__()
        self._future_steps = future_steps
        self._num_neighbors = num_neighbors
        self._num_modes = num_modes

        self.pedestrian_net = AgentEncoder()    # historia → embedding 256
        self.agent_agent = Agent2Agent(modes=num_modes)
        self.plan = AVDecoder(future_steps=self._future_steps, num_modes=num_modes)
        self.predict = AgentDecoder(future_steps=self._future_steps, num_neighbors=num_neighbors, num_modes=num_modes)
        self.score = Score(num_modes=num_modes)

    def forward(self, ego, neighbors):
        """
        ego:       (B, T_obs, feat_dim)              # peatón ego
        neighbors: (B, num_neighbors, T_obs, feat_dim)
                    feat_dim: 2 o 3 (x,y,(θ))
        """
        B = ego.shape[0]

        ego_embed = self.pedestrian_net(ego)  # (B, 256)

        neighbor_embeds = torch.stack(
            [self.pedestrian_net(neighbors[:, i]) for i in range(self._num_neighbors)],
            dim=1
        )  # (B, num_neighbors, 256)

        actors = torch.cat([ego_embed.unsqueeze(1), neighbor_embeds], dim=1)

        B = ego.shape[0]
        device = ego.device

        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B,1)
        neighbor_last = neighbors[:, :, -1, :]             # (B, N, feat_dim)
        neighbor_mask = (neighbor_last.sum(dim=-1) == 0)   # (B, N)  True = padding
        actor_mask = torch.cat([ego_mask, neighbor_mask], dim=1)  # (B, 1+N)
        agent_agent = self.agent_agent(actors, actor_mask)
        agent_agent = self.agent_agent(actors, actor_mask)
        ego_modes = agent_agent[:, :, 0, :]  
        plans, cost_function_weights = self.plan(ego_modes, None)
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
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
