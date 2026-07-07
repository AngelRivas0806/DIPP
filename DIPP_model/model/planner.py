import torch
import theseus as th
from utils.train_utils import bicycle_model  

# ================================
# u: (B, 2T) -> control: (B, T, 2)
# ================================
def _reshape_control(u: torch.Tensor):
    T = u.shape[-1] // 2
    return u.view(-1, T, 2), T

# =========================
""" Residual functions"""
# =========================

def speed_limit_residual(optim_vars, aux_vars):
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)

    current_state = aux_vars[0].tensor
    dt = aux_vars[1].tensor[0]
    max_speed = aux_vars[2].tensor[0]

    ego_current_state = current_state[:, 0]

    ego_traj = bicycle_model(control, ego_current_state, dt=dt)

    # bicycle_model devuelve [x, y, theta, v]
    speed = ego_traj[:, :, 3]

    speed_violation = torch.clamp(speed - max_speed, min=0.0)

    return speed_violation

def acceleration_residual(optim_vars, aux_vars):
    u = optim_vars[0].tensor                 # (B, 2T)
    control, T = _reshape_control(u)         # (B, T, 2)
    acc = control[:, :, 0]                   # (B, T)
    return acc

def jerk_residual(optim_vars, aux_vars):
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)
    acc = control[:, :, 0]                   # (B, T)
    dt = aux_vars[0].tensor[0]               # scalar tensor (no .item())
    return torch.diff(acc, dim=1) / dt       # (B, T-1)

def steering_residual(optim_vars, aux_vars):
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)
    steering = control[:, :, 1]              # (B, T)
    return steering

def steering_change_residual(optim_vars, aux_vars):
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)
    steering = control[:, :, 1]              # (B, T)
    dt = aux_vars[0].tensor[0]               # scalar tensor (no .item())
    return torch.diff(steering, dim=1) / dt  # (B, T-1)

def collision_avoidance_residual(optim_vars, aux_vars):
    """
    aux_vars:
      0: predictions  (B, K, T, 2)   vecinos predichos [x,y]
      1: current_state (B, 1+K, 8)   ego + vecinos
      2: dt (scalar)
      3: safety_distance (scalar)
    returns:
      (B, T)
    """
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)

    predictions = aux_vars[0].tensor         # (B, K, T, 2)
    current_state = aux_vars[1].tensor       # (B, 1+K, 8)
    safety_distance = aux_vars[3].tensor[0]  # scalar tensor (no .item())

    ego_current_state = current_state[:, 0]  # (B, 8)
    dt = aux_vars[2].tensor[0]  # dt_var (scalar tensor)
    ego_traj = bicycle_model(control, ego_current_state, dt=dt)
    ego_pos = ego_traj[:, :, :2]             # (B, T, 2)

    neighbors_pos = predictions[:, :, :, :2] # (B, K, T, 2)

    # máscara vecinos válidos (si todo ~0, lo tomamos como padding)
    neighbor_valid_mask = (neighbors_pos.abs().sum(dim=-1).sum(dim=-1) > 0.01)  # (B, K)

    # distancias ego->vecinos por timestep
    ego_pos_exp = ego_pos.unsqueeze(1)       # (B, 1, T, 2)
    distances = torch.norm(ego_pos_exp - neighbors_pos, dim=-1)  # (B, K, T)

    distances = torch.where(
        neighbor_valid_mask.unsqueeze(-1).expand_as(distances),
        distances,
        torch.ones_like(distances) * 1e4
    )

    min_distances, _ = torch.min(distances, dim=1)        # (B, T)
    collision = torch.clamp(safety_distance - min_distances, min=0.0)  # (B, T)

    # escala suave (opcional)
    return 0.1 * collision

def endpoint_goal_residual(optim_vars, aux_vars):
    """
    Costo de acercamiento al último punto del GT del ego.

    aux_vars:
      0: gt_trajectory (B, T, D)   contiene al menos [x, y]
      1: current_state (B, 1+K, 6/8)
      2: dt

    returns:
      (B, 2)  error final en x,y
    """

    u = optim_vars[0].tensor
    control, T = _reshape_control(u)

    gt_trajectory = aux_vars[0].tensor       # (B, T, D)
    current_state = aux_vars[1].tensor       # (B, 1+K, state_dim)
    dt = aux_vars[2].tensor[0]

    ego_current_state = current_state[:, 0]  # (B, state_dim)

    # Simulamos trayectoria del ego con los controles optimizados
    ego_traj = bicycle_model(control, ego_current_state, dt=dt)  # (B, T, 4)

    # Última posición planeada
    pred_final_pos = ego_traj[:, -1, :2]      # (B, 2)

    # Última posición del ground truth del ego
    gt_final_pos = gt_trajectory[:, -1, :2]   # (B, 2)

    # Residuo: diferencia final en x,y
    endpoint_error = pred_final_pos - gt_final_pos  # (B, 2)

    return 0.1 * endpoint_error


# =========================
# Objective builder
# =========================
def build_objective(
    objective: th.Objective, # Contenedor para agregar residuales, el optimizador minimiza la suma de estos.
    control_variables: th.Vector, # Variable de optimizacion (control), lo que Theseus ajusta
    current_state: th.Variable, # th.variable son variables auxiliares no optimizables
    predictions: th.Variable,
    gt_trajectory: th.Variable,
    weights: dict,
    trajectory_len: int,
    dt_var: th.Variable,
    safety_distance_var: th.Variable,
    max_speed_var: th.Variable,
    vectorize: bool = True,
    ):
    T = trajectory_len
    Tm1 = max(T - 1, 1)

    # Comfort
    objective.add(th.AutoDiffCostFunction([control_variables], acceleration_residual, T, weights["acc"], autograd_vectorize=vectorize, name="acceleration"))
    objective.add(th.AutoDiffCostFunction([control_variables], jerk_residual, Tm1, weights["jerk"], aux_vars=[dt_var], autograd_vectorize=vectorize, name="jerk"))
    objective.add(th.AutoDiffCostFunction([control_variables], steering_residual, T, weights["steer"], autograd_vectorize=vectorize, name="steering"))
    objective.add(th.AutoDiffCostFunction([control_variables], steering_change_residual, Tm1, weights["steer_change"], aux_vars=[dt_var], autograd_vectorize=vectorize, name="steering_change"))
    objective.add(th.AutoDiffCostFunction([control_variables], speed_limit_residual, T, weights["speed_limit"], aux_vars=[current_state, dt_var, max_speed_var], autograd_vectorize=vectorize,
        name="speed_limit"))
    objective.add(th.AutoDiffCostFunction([control_variables], collision_avoidance_residual, T, weights["collision"], aux_vars=[predictions, current_state, dt_var, safety_distance_var],
        autograd_vectorize=vectorize, name="collision_avoidance"))
    objective.add(th.AutoDiffCostFunction([control_variables], endpoint_goal_residual, 2, weights["endpoint_goal"], aux_vars=[gt_trajectory, current_state, dt_var],
            autograd_vectorize=vectorize, name="endpoint_goal"))

    return objective


# =========================
# MotionPlanner class
# =========================
class MotionPlanner:
    """
    Cost weights learnables (1 por costo):
      - acc
      - jerk
      - steer
      - steer_change
      - collision
      - endpoint_goal
    """
    def __init__(self, trajectory_len: int, device, test: bool = False, dt: float = 0.4, safety_distance: float = 0.5, max_speed: float = 1.5):

        self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        self.T = trajectory_len

        # Variables
        self.control_variables = th.Vector(dof=trajectory_len * 2, name="control_variables")
        self.predictions = th.Variable(torch.empty(1, 10, trajectory_len, 2), name="predictions")
        self.current_state = th.Variable(torch.empty(1, 11, 6), name="current_state")
        self.gt_trajectory = th.Variable(torch.empty(1, trajectory_len, 2, device=self.device), name="gt_trajectory")

        # Aux scalars as Variables
        self.dt_var = th.Variable(torch.tensor([dt]), name="dt")
        self.safety_distance_var = th.Variable(torch.tensor([safety_distance]), name="safety_distance")
        self.max_speed_var = th.Variable(torch.tensor([max_speed], dtype=torch.float32, device=self.device), name="max_speed")

        # Learnable weights (7)
        self.cost_weights_vars = {
            "acc":          th.Variable(torch.rand(1), name="w_acc"),
            "jerk":         th.Variable(torch.rand(1), name="w_jerk"),
            "steer":        th.Variable(torch.rand(1), name="w_steer"),
            "steer_change": th.Variable(torch.rand(1), name="w_steer_change"),
            "collision":    th.Variable(torch.rand(1), name="w_collision"),
            "speed_limit":  th.Variable(torch.rand(1), name="w_speed_limit"),
            "endpoint_goal": th.Variable(torch.rand(1), name="w_endpoint_goal"),
        }
        self.cost_weights = {k: th.ScaleCostWeight(v) for k, v in self.cost_weights_vars.items()}

        # Objective
        objective = th.Objective()

        self.objective = build_objective(
            objective=objective,
            control_variables=self.control_variables,
            current_state=self.current_state,
            predictions=self.predictions,
            gt_trajectory=self.gt_trajectory,
            weights=self.cost_weights,
            trajectory_len=trajectory_len,
            dt_var=self.dt_var,
            safety_distance_var=self.safety_distance_var,
            max_speed_var=self.max_speed_var,
            vectorize=False,
        )

        # Optimizer
        if test:
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=3, step_size=0.3, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=5, step_size=0.15, abs_err_tolerance=1e-3)

        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(self.device)  