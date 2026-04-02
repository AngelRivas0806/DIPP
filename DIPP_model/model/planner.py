import torch
import theseus as th
from utils.train_utils import bicycle_model  

# =========================
# Helpers
# =========================
def _reshape_control(u: torch.Tensor):
    """
    u: (B, 2T) -> control: (B, T, 2)
    """
    T = u.shape[-1] // 2
    return u.view(-1, T, 2), T


# =========================
# Residual functions
# =========================
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
    ego_traj = bicycle_model(control, ego_current_state)  # (B, T, 4) -> [x,y,theta,v]
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

def trajectory_following_residual(optim_vars, aux_vars):
    """
    aux_vars:
      0: gt_trajectory (B, T, 2)  [x, y]
      1: current_state (B, 1+K, 8)
    returns:
      (B, 2 * ceil(T/2))  (x errors + y errors sampled)
    """
    u = optim_vars[0].tensor
    control, T = _reshape_control(u)

    gt_trajectory = aux_vars[0].tensor       # (B, T, 3)
    current_state = aux_vars[1].tensor       # (B, 1+K, 8)

    ego_current_state = current_state[:, 0]  # (B, 8)
    ego_traj = bicycle_model(control, ego_current_state)  # (B, T, 4)
    pos_pred = ego_traj[:, :, :2]            # (B, T, 2)
    pos_gt = gt_trajectory[:, :, :2]         # (B, T, 2)

    # muestreo cada 2 steps (0,2,4,...)
    pos_pred_s = pos_pred[:, ::2, :]         # (B, ceil(T/2), 2)
    pos_gt_s = pos_gt[:, ::2, :]             # (B, ceil(T/2), 2)

    err = pos_pred_s - pos_gt_s              # (B, ceil(T/2), 2)
    err_x = err[:, :, 0]                     # (B, ceil(T/2))
    err_y = err[:, :, 1]                     # (B, ceil(T/2))

    traj_err = torch.cat([err_x, err_y], dim=1)  # (B, 2*ceil(T/2))
    return 0.1 * traj_err


# =========================
# Objective builder
# =========================
def build_objective(
    objective: th.Objective,
    control_variables: th.Vector,
    current_state: th.Variable,
    predictions: th.Variable,
    weights: dict,
    trajectory_len: int,
    dt_var: th.Variable,
    safety_distance_var: th.Variable,
    vectorize: bool = True,
):
    T = trajectory_len
    Tm1 = max(T - 1, 1)

    # Comfort
    objective.add(th.AutoDiffCostFunction(
        [control_variables], acceleration_residual, T,
        weights["acc"], autograd_vectorize=vectorize, name="acceleration"
    ))
    objective.add(th.AutoDiffCostFunction(
        [control_variables], jerk_residual, Tm1,
        weights["jerk"], aux_vars=[dt_var], autograd_vectorize=vectorize, name="jerk"
    ))
    objective.add(th.AutoDiffCostFunction(
        [control_variables], steering_residual, T,
        weights["steer"], autograd_vectorize=vectorize, name="steering"
    ))
    objective.add(th.AutoDiffCostFunction(
        [control_variables], steering_change_residual, Tm1,
        weights["steer_change"], aux_vars=[dt_var], autograd_vectorize=vectorize, name="steering_change"
    ))

    # Safety
    objective.add(th.AutoDiffCostFunction(
        [control_variables], collision_avoidance_residual, T,
        weights["collision"],
        aux_vars=[predictions, current_state, dt_var, safety_distance_var],
        autograd_vectorize=vectorize,
        name="collision_avoidance"
    ))

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
    """
    def __init__(self, trajectory_len: int, device, test: bool = False, dt: float = 0.1, safety_distance: float = 0.5):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)
        self.T = trajectory_len

        # Variables
        self.control_variables = th.Vector(dof=trajectory_len * 2, name="control_variables")
        self.predictions = th.Variable(torch.empty(1, 10, trajectory_len, 2), name="predictions")
        self.current_state = th.Variable(torch.empty(1, 11, 6), name="current_state")

        # Aux scalars as Variables
        self.dt_var = th.Variable(torch.tensor([dt]), name="dt")
        self.safety_distance_var = th.Variable(torch.tensor([safety_distance]), name="safety_distance")

        # Learnable weights (5 costos, sin tracking)
        self.cost_weights_vars = {
            "acc":          th.Variable(torch.rand(1), name="w_acc"),
            "jerk":         th.Variable(torch.rand(1), name="w_jerk"),
            "steer":        th.Variable(torch.rand(1), name="w_steer"),
            "steer_change": th.Variable(torch.rand(1), name="w_steer_change"),
            "collision":    th.Variable(torch.rand(1), name="w_collision"),
        }
        self.cost_weights = {k: th.ScaleCostWeight(v) for k, v in self.cost_weights_vars.items()}

        # Objective
        objective = th.Objective()
        self.objective = build_objective(
            objective=objective,
            control_variables=self.control_variables,
            current_state=self.current_state,
            predictions=self.predictions,
            weights=self.cost_weights,
            trajectory_len=trajectory_len,
            dt_var=self.dt_var,
            safety_distance_var=self.safety_distance_var,
            vectorize=False,
        )

        # Optimizer
        if test:
            self.optimizer = th.GaussNewton(
                objective, th.CholeskyDenseSolver,
                vectorize=False, max_iterations=50, step_size=0.3, abs_err_tolerance=1e-2
            )
        else:
            self.optimizer = th.GaussNewton(
                objective, th.LUDenseSolver,
                vectorize=False, max_iterations=10, step_size=0.3
            )

        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(self.device)   # <-- no encadenes