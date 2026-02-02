"""
Predictor simplificado para modo sin ego.
Solo predice trayectorias futuras basado en trayectorias observadas.
No hay agente ego, no hay acciones de control.
"""

import torch
from torch import nn
import torch.nn.functional as F

# Número de modos de predicción
NUM_MODES = 20


class TrajectoryEncoder(nn.Module):
    """Codifica una trayectoria histórica usando LSTM."""
    
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=2):
        super(TrajectoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, trajectory):
        """
        Args:
            trajectory: (batch, seq_len, 2) - posiciones (x, y)
        Returns:
            embedding: (batch, 256) - representación codificada
        """
        output, _ = self.lstm(trajectory)
        return output[:, -1]  # último estado oculto


class MultiModalTransformer(nn.Module):
    """Genera múltiples modos de predicción usando atención multi-cabeza."""
    
    def __init__(self, modes=NUM_MODES, feature_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, 4, 0.1, batch_first=True) 
            for _ in range(modes)
        ])
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, query, key, value):
        """
        Args:
            query, key, value: (batch, 1, 256) - embeddings
        Returns:
            output: (batch, num_modes, 256) - embeddings multimodales
        """
        attention_outputs = []
        for i in range(self.modes):
            attn_out, _ = self.attention[i](query, key, value)
            attention_outputs.append(attn_out)
        
        attention_output = torch.stack(attention_outputs, dim=1)  # (batch, modes, 1, 256)
        attention_output = attention_output.squeeze(2)  # (batch, modes, 256)
        output = self.ffn(attention_output)
        
        return output


class TrajectoryDecoder(nn.Module):
    """Decodifica embeddings en trayectorias futuras."""
    
    def __init__(self, future_steps=12, feature_dim=256, num_modes=NUM_MODES):
        super(TrajectoryDecoder, self).__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        
        self.decode = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, future_steps * 2)  # 2 = (x, y)
        )
    
    def forward(self, embeddings, last_position):
        """
        Args:
            embeddings: (batch, num_modes, 256)
            last_position: (batch, 2) - última posición observada
        Returns:
            trajectories: (batch, num_modes, future_steps, 2)
        """
        batch_size = embeddings.shape[0]
        
        # Decodificar desplazamientos
        deltas = self.decode(embeddings)  # (batch, num_modes, future_steps*2)
        deltas = deltas.view(batch_size, self.num_modes, self.future_steps, 2)
        
        # Calcular posiciones absolutas desde última posición observada
        trajectories = deltas + last_position.unsqueeze(1).unsqueeze(2)
        
        return trajectories


class ModeScorer(nn.Module):
    """Asigna probabilidades a cada modo de predicción."""
    
    def __init__(self, feature_dim=256, num_modes=NUM_MODES):
        super(ModeScorer, self).__init__()
        self.num_modes = num_modes
        
        self.score_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, num_modes, 256)
        Returns:
            scores: (batch, num_modes) - probabilidades de cada modo
        """
        scores = self.score_net(embeddings).squeeze(-1)  # (batch, num_modes)
        return scores


class PredictorNoEgo(nn.Module):
    """
    Predictor de trayectorias sin agente ego.
    Entrada: trayectoria observada (x, y)
    Salida: trayectoria futura predicha (x, y) con múltiples modos
    """
    
    def __init__(self, obs_len=8, future_steps=12, num_modes=NUM_MODES):
        super(PredictorNoEgo, self).__init__()
        self.obs_len = obs_len
        self.future_steps = future_steps
        self.num_modes = num_modes
        
        # Componentes del modelo
        self.encoder = TrajectoryEncoder(input_dim=2, hidden_dim=256, num_layers=2)
        self.multimodal = MultiModalTransformer(modes=num_modes, feature_dim=256)
        self.decoder = TrajectoryDecoder(future_steps=future_steps, num_modes=num_modes)
        self.scorer = ModeScorer(num_modes=num_modes)
    
    def forward(self, observed_trajectory):
        """
        Args:
            observed_trajectory: (batch, obs_len, 2) - posiciones observadas
        Returns:
            predictions: (batch, num_modes, future_steps, 2) - trayectorias futuras
            scores: (batch, num_modes) - probabilidad de cada modo
        """
        batch_size = observed_trajectory.shape[0]
        
        # 1. Codificar trayectoria observada
        embedding = self.encoder(observed_trajectory)  # (batch, 256)
        
        # 2. Generar múltiples modos
        embedding_expanded = embedding.unsqueeze(1)  # (batch, 1, 256)
        multimodal_embeddings = self.multimodal(
            embedding_expanded, 
            embedding_expanded, 
            embedding_expanded
        )  # (batch, num_modes, 256)
        
        # 3. Decodificar trayectorias futuras
        last_position = observed_trajectory[:, -1, :]  # (batch, 2)
        predictions = self.decoder(multimodal_embeddings, last_position)
        # predictions: (batch, num_modes, future_steps, 2)
        
        # 4. Calcular scores para cada modo
        scores = self.scorer(multimodal_embeddings)  # (batch, num_modes)
        
        return predictions, scores


if __name__ == "__main__":
    # Test del modelo
    model = PredictorNoEgo(obs_len=8, future_steps=8, num_modes=20)
    print(model)
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    batch_size = 32
    obs_traj = torch.randn(batch_size, 8, 2)
    predictions, scores = model(obs_traj)
    
    print(f'\nInput shape: {obs_traj.shape}')
    print(f'Predictions shape: {predictions.shape}')
    print(f'Scores shape: {scores.shape}')
    print(f'\nExpected:')
    print(f'  Predictions: (batch={batch_size}, modes=20, future_steps=12, features=2)')
    print(f'  Scores: (batch={batch_size}, modes=20)')
