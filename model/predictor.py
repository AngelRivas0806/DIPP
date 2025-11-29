import torch
from torch import nn
import torch.nn.functional as F

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

"""Local context encoders
In our case, itsn't necessary the classes because are relation with map elements"""

# class LaneEncoder(nn.Module):
#     def __init__(self):
#         super(LaneEncoder, self).__init__()
#         # encdoer layer
#         self.self_line = nn.Linear(3, 128)
#         self.left_line = nn.Linear(3, 128)
#         self.right_line = nn.Linear(3, 128)
#         self.speed_limit = nn.Linear(1, 64)
#         self.self_type = nn.Embedding(4, 64, padding_idx=0)
#         self.left_type = nn.Embedding(11, 64, padding_idx=0)
#         self.right_type = nn.Embedding(11, 64, padding_idx=0)
#         self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
#         self.interpolating = nn.Embedding(2, 64)
#         self.stop_sign = nn.Embedding(2, 64)
#         self.stop_point = nn.Embedding(2, 64)

#         # hidden layers
#         self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())

#     def forward(self, inputs):
#         # embedding
#         self_line = self.self_line(inputs[..., :3])
#         left_line = self.left_line(inputs[..., 3:6])
#         right_line = self.right_line(inputs[...,  6:9])
#         speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
#         self_type = self.self_type(inputs[..., 10].int())
#         left_type = self.left_type(inputs[..., 11].int())
#         right_type = self.right_type(inputs[..., 12].int()) 
#         traffic_light = self.traffic_light_type(inputs[..., 13].int())
#         stop_point = self.stop_point(inputs[..., 14].int())
#         interpolating = self.interpolating(inputs[..., 15].int()) 
#         stop_sign = self.stop_sign(inputs[..., 16].int())

#         lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
#         lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
#         # process
#         output = self.pointnet(lane_embedding)

#         return output

# class CrosswalkEncoder(nn.Module):
    # def __init__(self):
    #     super(CrosswalkEncoder, self).__init__()
    #     self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
    
    # def forward(self, inputs):
    #     output = self.point_net(inputs)

    #     return output

"""Transformer modules"""
# Cross-attention and multi-modal attention modules

# class CrossTransformer(nn.Module):
    # def __init__(self):
    #     super(CrossTransformer, self).__init__()
    #     self.cross_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)
    #     self.transformer = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 256), nn.LayerNorm(256))

    # def forward(self, query, key, value, mask=None):
    #     attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
    #     output = self.transformer(attention_output)

    #     return output
"""Multi-modal transformer module is used for agen2map, so we commented that code , we modified agent2agent to use it as well"""
class MultiModalTransformer(nn.Module):
    def __init__(self, modes=3, output_dim=256):
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

# Transformer-based encoders
# Compute auto-attention among agents.
# It has as input a tensor of shape (batch_size, num_agents, 256) and an optional mask of shape (batch_size, num_agents,256).the mask 
# indicates which agents should be ignored during attention computation (e.g., padding agents).

"""Original Agent2Agent, below we have the modified version using multi-modal attention"""
# class Agent2Agent(nn.Module):
    # def __init__(self):
    #     super(Agent2Agent, self).__init__()
    #     encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
    #     self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # def forward(self, inputs, mask=None):
    #     output = self.interaction_net(inputs, src_key_padding_mask=mask)
    #         return output



class Agent2Agent(nn.Module):
    def __init__(self, modes=3):
        super(Agent2Agent, self).__init__()
        self.modes = modes
        
        # Encoder to process interactions between agents
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu', batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Multi-modal attention to generate 3 interpretations
        self.multimodal = MultiModalTransformer(modes=3, output_dim=256)
        
    def forward(self, inputs, mask=None):
        # Step 1: Encode basic interactions
        encoded = self.interaction_net(inputs, src_key_padding_mask=mask)
        # encoded: (batch, num_agents, 256)

        # Step 2: Generate 3 modes using self-attention
        query = encoded  # (batch, num_agents, 256)
        output = self.multimodal(query, encoded, encoded, mask)
        # output: (batch, 3, num_agents, 256)
        
        return output
    

"""Agen2map isn't neccesary so we commented that code""" 
# class Agent2Map(nn.Module):
    # def __init__(self):
    #     super(Agent2Map, self).__init__()
    #     self.lane_attention = CrossTransformer()
    #     self.crosswalk_attention = CrossTransformer()
    #     self.map_attention = MultiModalTransformer() 

    # def forward(self, actor, lanes, crosswalks, mask):
    #     query = actor.unsqueeze(1)
    #     lanes_actor = [self.lane_attention(query, lanes[:, i], lanes[:, i]) for i in range(lanes.shape[1])]
    #     crosswalks_actor = [self.crosswalk_attention(query, crosswalks[:, i], crosswalks[:, i]) for i in range(crosswalks.shape[1])]
    #     map_actor = torch.cat(lanes_actor+crosswalks_actor, dim=1)
    #     output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

    #     return map_actor, output 

"""Decoders"""

class AgentDecoder(nn.Module):
    def __init__(self, future_steps):
        super(AgentDecoder, self).__init__()
        self._future_steps = future_steps 
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, future_steps*3))

    def transform(self, prediction, current_state):
        x = current_state[:, 0] 
        y = current_state[:, 1]
        theta = current_state[:, 2]
        delta_x = prediction[:, :, 0]
        delta_y = prediction[:, :, 1]
        delta_theta = prediction[:, :, 2]
        new_x = x.unsqueeze(1) + delta_x 
        new_y = y.unsqueeze(1) + delta_y 
        new_theta = theta.unsqueeze(1) + delta_theta
        traj = torch.stack([new_x, new_y, new_theta], dim=-1)

        return traj
       
    def forward(self, agent_agent, current_state):
        decoded = self.decode(agent_agent).view(-1, 3, 10, self._future_steps, 3)
        trajs = torch.stack([self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(10)], dim=1)
        trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))

        return trajs

"""Only change the dimensions of the tensors, because we aren't concatenating map features anymore"""
# class AVDecoder(nn.Module):
    # def __init__(self, future_steps=50, feature_len=9):
    #     super(AVDecoder, self).__init__()
    #     self._future_steps = future_steps
    #     self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
    #     self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
    #     self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
    #     self.register_buffer('constraint', torch.tensor([[10, 10]]))

    # def forward(self, agent_map, agent_agent):
    #     feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
    #     actions = self.control(feature).view(-1, 3, self._future_steps, 2)
    #     dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
    #     cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

    #     return actions, cost_function_weights

class AVDecoder(nn.Module):
    def __init__(self, future_steps=12, feature_len=9):
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        # Output control variables (acceleration and steering) instead of trajectories
        self.control = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_agent, current_state):
        # Generate control variables for 3 modes
        actions = self.control(agent_agent).view(-1, 3, self._future_steps, 2)
        
        # Generate cost function weights
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return actions, cost_function_weights

"""Score module"""
# class Score(nn.Module):
    # def __init__(self):
    #     super(Score, self).__init__()
    #     self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
    #     self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1))

    # def forward(self, map_feature, agent_agent, agent_map):
    #     # pooling
    #     map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
    #     map_feature = torch.max(map_feature, dim=1)[0]
    #     agent_agent = torch.max(agent_agent, dim=1)[0]
    #     agent_map = torch.max(agent_map, dim=2)[0]

    #     feature = torch.cat([map_feature, agent_agent], dim=-1)
    #     feature = self.reduce(feature.detach())
    #     feature = torch.cat([feature.unsqueeze(1).repeat(1, 3, 1), agent_map.detach()], dim=-1)
    #     scores = self.decode(feature).squeeze(-1)

    #     return scores

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, agent_agent):
        # agent_agent: (batch, 3, num_agents, 256)
        # Pooling over all agents for each mode
        agent_pooled = torch.max(agent_agent, dim=2)[0]  # (batch, 3, 256)
        scores = self.decode(agent_pooled).squeeze(-1)  # (batch, 3)
        # scores → (batch, 3)
        return scores

# Build predictor
# class Predictor(nn.Module):
#     def __init__(self, future_steps):
#         super(Predictor, self).__init__()
#         self._future_steps = future_steps

#         # agent layer
#         # self.vehicle_net = AgentEncoder()
#         self.pedestrian_net = AgentEncoder()
#         # self.cyclist_net = AgentEncoder()

#         # map layer
#         self.lane_net = LaneEncoder()
#         self.crosswalk_net = CrosswalkEncoder()
        
#         # attention layers
#         self.agent_map = Agent2Map()
#         self.agent_agent = Agent2Agent()

#         # decode layers
#         self.plan = AVDecoder(self._future_steps)
#         self.predict = AgentDecoder(self._future_steps)
#         self.score = Score()

#     def forward(self, ego, neighbors, map_lanes, map_crosswalks):
#         # actors
#         ego_actor = self.vehicle_net(ego)
#         vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
#         pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
#         cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
#         neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
#         neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
#         actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
#         actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]

#         # maps
#         lane_feature = self.lane_net(map_lanes)
#         crosswalk_feature = self.crosswalk_net(map_crosswalks)
#         lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
#         crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
#         map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
#         map_mask[:, :, 0] = False # prevent nan
        
#         # actor to actor
#         agent_agent = self.agent_agent(actors, actor_mask)
        
#         # map to actor
#         map_feature = []
#         agent_map = []
#         for i in range(actors.shape[1]):
#             output = self.agent_map(agent_agent[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
#             map_feature.append(output[0])
#             agent_map.append(output[1])

#         map_feature = torch.stack(map_feature, dim=1)
#         agent_map = torch.stack(agent_map, dim=2)

#         # plan + prediction 
#         plans, cost_function_weights = self.plan(agent_map[:, :, 0], agent_agent[:, 0])
#         predictions = self.predict(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
#         scores = self.score(map_feature, agent_agent, agent_map)
        
#         return plans, predictions, scores, cost_function_weights

# class Predictor(nn.Module):
#     def __init__(self, future_steps):
#         super(Predictor, self).__init__()
#         self._future_steps = future_steps

#         # agent encoder
#         self.pedestrian_net = AgentEncoder()
        
#         # attention layer
#         self.agent_agent = Agent2Agent()

#         # decode layers
#         self.plan = AVDecoder(self._future_steps)
#         self.predict = AgentDecoder(self._future_steps)
#         self.score = Score()
        
#         # Initialize weights
#         self._init_weights()
    
#     def _init_weights(self):
#         """Initialize model weights for stable training"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LSTM):
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         nn.init.xavier_uniform_(param.data)
#                     elif 'weight_hh' in name:
#                         nn.init.orthogonal_(param.data)
#                     elif 'bias' in name:
#                         nn.init.constant_(param.data, 0)

#     def forward(self, ego, neighbors):
#         # Encode all pedestrians
#         ego_actor = self.pedestrian_net(ego)
#         neighbor_actors = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1)
#         actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        
#         # Create mask for padding
#         actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]
        
#         # Agent-to-agent attention (multi-modal)
#         agent_agent = self.agent_agent(actors, actor_mask)
#         # agent_agent shape: (batch, 3, 11, 256)
        
#         # Decode trajectories
#         current_state_ego = ego[:, -1, :3]
#         current_state_neighbors = neighbors[:, :, -1, :3]
        
#         plans, cost_function_weights = self.plan(agent_agent[:, :, 0], current_state_ego)  # (batch, 3, future_steps, 3)
#         predictions = self.predict(agent_agent[:, :, 1:], current_state_neighbors)  # (batch, 3, 10, future_steps, 3)
#         scores = self.score(agent_agent)  # (batch, 3)
        
#         return plans, predictions, scores, cost_function_weights

class Predictor(nn.Module):
    def __init__(self, future_steps, num_neighbors=10, num_modes=3):
        super(Predictor, self).__init__()
        self._future_steps = future_steps
        self._num_neighbors = num_neighbors
        self._num_modes = num_modes

        # Encoder compartido para todos los peatones (ego + vecinos)
        self.pedestrian_net = AgentEncoder()    # historia → embedding 256

        # Interacciones agente–agente (solo peatones)
        self.agent_agent = Agent2Agent(modes=num_modes)

        # Planner tipo DIPP para el ego (usa solo info social ahora)
        self.plan = AVDecoder(future_steps=self._future_steps)

        # Decoder de trayectorias futuras para los vecinos
        self.predict = AgentDecoder(future_steps=self._future_steps)

        # Score multimodal (tu versión sin mapa)
        self.score = Score()

    def forward(self, ego, neighbors):
        """
        ego:       (B, T_obs, feat_dim)              # peatón ego
        neighbors: (B, num_neighbors, T_obs, feat_dim)
                    feat_dim: 2 o 3 (x,y,(θ))
        """

        B = ego.shape[0]

        # 1) Encoder de ego y vecinos
        ego_embed = self.pedestrian_net(ego)  # (B, 256)

        neighbor_embeds = torch.stack(
            [self.pedestrian_net(neighbors[:, i]) for i in range(self._num_neighbors)],
            dim=1
        )  # (B, num_neighbors, 256)

        # 2) Construir lista de actores: [ego, vecino1, ..., vecinoN]
        actors = torch.cat([ego_embed.unsqueeze(1), neighbor_embeds], dim=1)
        # actors: (B, 1 + num_neighbors, 256) = (B, num_agents, 256)

        B = ego.shape[0]
        device = ego.device

        # 1) El ego NUNCA es padding
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # (B,1)

        # 2) Para los vecinos, miramos solo su último frame
        # neighbors: (B, num_neighbors, T_obs, feat_dim)
        neighbor_last = neighbors[:, :, -1, :]             # (B, N, feat_dim)
        neighbor_mask = (neighbor_last.sum(dim=-1) == 0)   # (B, N)  True = padding

        # 3) Juntamos ego + vecinos en una sola máscara
        actor_mask = torch.cat([ego_mask, neighbor_mask], dim=1)  # (B, 1+N)

        # 4) Interacciones agente–agente + multimodal
        # agent_agent: (B, num_modes, num_agents, 256)
        agent_agent = self.agent_agent(actors, actor_mask)

        # 5) Planner para el ego (tipo DIPP, pero sin mapa)
        # Tomamos solo el embedding del ego (agente 0) para cada modo
        # ego_modes: (B, num_modes, 256)
        ego_modes = agent_agent[:, :, 0, :]  

        # AVDecoder forward: actions, cost_function_weights
        # actions: (B, 3, future_steps, 2)
        # cost_function_weights: (1, 9) en tu implementación actual
        plans, cost_function_weights = self.plan(ego_modes, None)

        # 6) Estado actual de cada vecino (último frame observado)
        # Ej: si feat_dim=3 → (x,y,θ)
        current_state_neighbors = neighbors[:, :, -1, :]  # (B, num_neighbors, feat_dim)

        # 7) Predicción de trayectorias futuras de los vecinos
        # Pasamos solo los vecinos (omitimos agente 0 = ego)
        # agent_agent[:, :, 1:] → (B, num_modes, num_neighbors, 256)
        predictions = self.predict(
            agent_agent[:, :, 1:],      # embeddings por modo de vecinos
            current_state_neighbors     # estado actual de vecinos
        )
        # predictions: (B, num_modes, num_neighbors, future_steps, 3)

        # 8) Score por modo (solo interacciones sociales)
        scores = self.score(agent_agent)  # (B, num_modes) = (B, 3)

        # 9) Salida final (mismo "shape lógico" que tu versión original)
        return plans, predictions, scores, cost_function_weights


if __name__ == "__main__":
    # set up model
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
