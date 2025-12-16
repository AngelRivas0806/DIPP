import torch
import theseus as th
from utils.train_utils import project_to_frenet_frame

class MotionPlanner:
    def __init__(self, trajectory_len, feature_len, device, test=False):
        self.device = device

        # define cost function
        cost_function_weights = [th.ScaleCostWeight(th.Variable(torch.rand(1), name=f'cost_function_weight_{i+1}')) for i in range(feature_len)]
            
        # define control variable (trajectory_len * 2 for acceleration and steering)
        control_variables = th.Vector(dof=trajectory_len*2, name="control_variables")
        
        # define prediction variable (neighbors' predicted trajectories)
        predictions = th.Variable(torch.empty(1, 10, trajectory_len, 3), name="predictions")
        
        # define current state (ego + neighbors)
        current_state = th.Variable(torch.empty(1, 11, 8), name="current_state")
        
        # define ground truth trajectory for trajectory following cost
        gt_trajectory = th.Variable(torch.empty(1, trajectory_len, 3), name="gt_trajectory")

        # set up objective
        objective = th.Objective()
        self.objective = cost_function(objective, control_variables, current_state, predictions, gt_trajectory, cost_function_weights, trajectory_len)

        # set up optimizer
        if test:
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(objective, th.LUDenseSolver, vectorize=False, max_iterations=2, step_size=0.4)

        # set up motion planner
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)

# model
def bicycle_model(control, current_state):
    dt = 0.1 # discrete time period [s]
    max_a = 5 # vehicle's accleration limits [m/s^2]
    max_d = 0.5 # vehicle's steering limits [rad]
    L = 3.089 # vehicle's wheelbase [m]
    
    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_d, max_d) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta / L # use delta to approximate tan(delta)
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

# cost functions
def acceleration(optim_vars, aux_vars):
    # Dynamically compute trajectory length from tensor size
    traj_len = optim_vars[0].tensor.shape[0] // optim_vars[0].tensor.shape[0] # Will be overridden below
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    acc = control[:, :, 0]
    
    return acc

def jerk(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    acc = control[:, :, 0]
    jerk = torch.diff(acc) / 0.1
    
    return jerk

def steering(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    steering = control[:, :, 1]

    return steering 

def steering_change(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    steering = control[:, :, 1]
    steering_change = torch.diff(steering) / 0.1

    return steering_change


# ============================================================================
# NEW COST FUNCTIONS FOR PEDESTRIAN TRAJECTORY PLANNING
# ============================================================================

def collision_avoidance(optim_vars, aux_vars):
    """
    Penaliza la cercanía a otros peatones para evitar colisiones.
    
    Args:
        optim_vars[0]: control_variables [batch, 24] -> reshape to [batch, 12, 2]
        aux_vars[0]: predictions - trayectorias predichas de vecinos [batch, 10, 12, 3]
        aux_vars[1]: current_state - estado actual [batch, 11, 8]
    
    Returns:
        collision_error: penalización por cercanía [batch, 12]
    """
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    predictions = aux_vars[0].tensor  # [batch, 10, 12, 3] - vecinos predichos
    current_state = aux_vars[1].tensor  # [batch, 11, 8]
    
    # Generar trayectoria del ego usando bicycle model
    ego_current_state = current_state[:, 0]  # [batch, 8]
    ego_traj = bicycle_model(control, ego_current_state)  # [batch, 12, 4] -> [x, y, theta, v]
    
    # Extraer solo posiciones del ego
    ego_pos = ego_traj[:, :, :2]  # [batch, 12, 2]
    
    # Extraer posiciones de vecinos
    neighbors_pos = predictions[:, :, :, :2]  # [batch, 10, 12, 2]
    
    # Crear máscara para vecinos válidos (no zeros/padding)
    # Un vecino es válido si su posición no es todo ceros
    neighbor_valid_mask = (neighbors_pos.abs().sum(dim=-1).sum(dim=-1) > 0.01)  # [batch, 10]
    
    # Distancia mínima de seguridad para peatones (metros)
    safety_distance = 0.5  # Reducido a 50 cm para ser menos agresivo
    
    # Calcular distancias a cada vecino en cada timestep
    ego_pos_expanded = ego_pos.unsqueeze(1)  # [batch, 1, 12, 2]
    
    # Distancia euclidiana a cada vecino
    distances = torch.norm(ego_pos_expanded - neighbors_pos, dim=-1)  # [batch, 10, 12]
    
    # Poner distancia muy grande para vecinos inválidos (padding)
    distances = torch.where(
        neighbor_valid_mask.unsqueeze(-1).expand_as(distances),
        distances,
        torch.ones_like(distances) * 100.0  # Distancia grande para padding
    )
    
    # Encontrar la distancia mínima a cualquier vecino válido en cada timestep
    min_distances, _ = torch.min(distances, dim=1)  # [batch, 12]
    
    # Penalización: mayor cuando la distancia es menor que safety_distance
    collision_error = torch.clamp(safety_distance - min_distances, min=0)  # [batch, 12]
    
    # Escalar para que el error sea más pequeño (evitar gradientes explosivos)
    collision_error = collision_error * 0.1
    
    return collision_error


def goal_reaching(optim_vars, aux_vars):
    """
    Penaliza la distancia a la meta (última posición del ground truth).
    Ayuda a que la trayectoria planificada se dirija hacia el destino.
    
    Args:
        optim_vars[0]: control_variables [batch, 24]
        aux_vars[0]: gt_trajectory - ground truth [batch, 12, 3]
        aux_vars[1]: current_state - estado actual [batch, 11, 8]
    
    Returns:
        goal_error: distancia a la meta en el último timestep [batch, 2]
    """
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    gt_trajectory = aux_vars[0].tensor  # [batch, 12, 3]
    current_state = aux_vars[1].tensor  # [batch, 11, 8]
    
    # Generar trayectoria del ego
    ego_current_state = current_state[:, 0]  # [batch, 8]
    ego_traj = bicycle_model(control, ego_current_state)  # [batch, 12, 4]
    
    # Posición final predicha
    final_pos_pred = ego_traj[:, -1, :2]  # [batch, 2]
    
    # Posición final del ground truth (la meta)
    final_pos_gt = gt_trajectory[:, -1, :2]  # [batch, 2]
    
    # Error: diferencia entre posición final predicha y la meta
    goal_error = final_pos_pred - final_pos_gt  # [batch, 2]
    
    # Escalar para evitar gradientes explosivos
    goal_error = goal_error * 0.1
    
    return goal_error


def trajectory_following(optim_vars, aux_vars):
    """
    Penaliza la desviación de la trayectoria ground truth.
    Ayuda a que la trayectoria planificada siga de cerca la trayectoria real.
    
    Args:
        optim_vars[0]: control_variables [batch, 24]
        aux_vars[0]: gt_trajectory - ground truth [batch, 12, 3]
        aux_vars[1]: current_state - estado actual [batch, 11, 8]
    
    Returns:
        traj_error: error de posición en cada timestep [batch, 12] (6 errores x + 6 errores y)
    """
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    gt_trajectory = aux_vars[0].tensor  # [batch, 12, 3]
    current_state = aux_vars[1].tensor  # [batch, 11, 8]
    
    # Generar trayectoria del ego
    ego_current_state = current_state[:, 0]  # [batch, 8]
    ego_traj = bicycle_model(control, ego_current_state)  # [batch, 12, 4]
    
    # Posiciones predichas
    pos_pred = ego_traj[:, :, :2]  # [batch, 12, 2]
    
    # Posiciones ground truth
    pos_gt = gt_trajectory[:, :, :2]  # [batch, 12, 2]
    
    # Error de seguimiento (muestreamos cada 2 timesteps para reducir dimensión)
    traj_error_x = pos_pred[:, ::2, 0] - pos_gt[:, ::2, 0]  # [batch, 6]
    traj_error_y = pos_pred[:, ::2, 1] - pos_gt[:, ::2, 1]  # [batch, 6]
    
    # Concatenar errores x e y
    traj_error = torch.cat([traj_error_x, traj_error_y], dim=-1)  # [batch, 12]
    
    # Escalar para evitar gradientes explosivos
    traj_error = traj_error * 0.1
    
    return traj_error


def speed(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    current_state = aux_vars[1].tensor[:, 0]
    velocity = torch.hypot(current_state[:, 3], current_state[:, 4]) 
    dt = 0.1

    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(aux_vars[0].tensor[:, :, -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit

    return speed_error

def lane_xy(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]
    
    traj = bicycle_model(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)

    return lane_error

def lane_theta(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]

    traj = bicycle_model(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    theta = traj[:, :, 2]
    lane_error = theta[:, 1::2] - ref_points[:, 1::2, 2]
    
    return lane_error

def red_light_violation(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    current_state = aux_vars[1].tensor[:, 0]
    ref_line = aux_vars[0].tensor
    red_light = ref_line[..., -1]
    dt = 0.1

    velocity = torch.hypot(current_state[:, 3], current_state[:, 4]) 
    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    s = torch.cumsum(speed * dt, dim=-1)

    stop_point = torch.max(red_light[:, 200:]==0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1) - 3
    red_light_error = (s - stop_distance) * (s > stop_distance) * (stop_point.unsqueeze(-1) != 0)

    return red_light_error

def safety(optim_vars, aux_vars):
    control = optim_vars[0].tensor.reshape(-1, 12, 2)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego = bicycle_model(control, ego_current_state)
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(ego.detach(), ref_line)
    
    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]: # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(ego[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=1)

    return safe_error

def cost_function(objective, control_variables, current_state, predictions, gt_trajectory, cost_function_weights, trajectory_len, vectorize=True):
    """
    Combina todas las funciones de costo con sus pesos.
    
    Mapeo de pesos (todos los 9 pesos deben estar asociados):
    - weight[0]: acceleration_aux (duplicado para usar este peso)
    - weight[1]: acceleration
    - weight[2]: jerk  
    - weight[3]: steering
    - weight[4]: steering_change
    - weight[5]: collision_avoidance (NUEVO)
    - weight[6]: goal_reaching (NUEVO)
    - weight[7]: trajectory_following (NUEVO)
    - weight[8]: goal_reaching_aux (duplicado para usar este peso)
    """
    
    # ========== COMFORT COSTS (suavidad del movimiento) ==========
    
    # Penaliza aceleraciones altas (peso principal)
    acc_cost = th.AutoDiffCostFunction(
        [control_variables], 
        acceleration, 
        trajectory_len, 
        cost_function_weights[1], 
        autograd_vectorize=vectorize, 
        name="acceleration"
    )
    objective.add(acc_cost)
    
    # Peso 0: usar para acceleration también (para evitar warning)
    acc_cost_0 = th.AutoDiffCostFunction(
        [control_variables], 
        acceleration, 
        trajectory_len, 
        cost_function_weights[0],  # Asocia weight[0]
        autograd_vectorize=vectorize, 
        name="acceleration_aux"
    )
    objective.add(acc_cost_0)
    
    # Penaliza cambios bruscos de aceleración (jerk)
    jerk_cost = th.AutoDiffCostFunction(
        [control_variables], 
        jerk, 
        trajectory_len-1, 
        cost_function_weights[2], 
        autograd_vectorize=vectorize, 
        name="jerk"
    )
    objective.add(jerk_cost)
    
    # Penaliza ángulos de dirección grandes
    steering_cost = th.AutoDiffCostFunction(
        [control_variables], 
        steering, 
        trajectory_len, 
        cost_function_weights[3], 
        autograd_vectorize=vectorize, 
        name="steering"
    )
    objective.add(steering_cost)
    
    # Penaliza cambios bruscos de dirección
    steering_change_cost = th.AutoDiffCostFunction(
        [control_variables], 
        steering_change, 
        trajectory_len-1, 
        cost_function_weights[4], 
        autograd_vectorize=vectorize, 
        name="steering_change"
    )
    objective.add(steering_change_cost)
    
    # ========== SAFETY COSTS (evitar colisiones) ==========
    
    # Penaliza cercanía a otros peatones
    collision_cost = th.AutoDiffCostFunction(
        [control_variables], 
        collision_avoidance, 
        trajectory_len,  # 12 valores de error (uno por timestep)
        cost_function_weights[5], 
        aux_vars=[predictions, current_state], 
        autograd_vectorize=vectorize, 
        name="collision_avoidance"
    )
    objective.add(collision_cost)
    
    # ========== GOAL COSTS (alcanzar destino) ==========
    
    # Penaliza distancia a la meta final
    goal_cost = th.AutoDiffCostFunction(
        [control_variables], 
        goal_reaching, 
        2,  # 2 valores: error en x, error en y
        cost_function_weights[6], 
        aux_vars=[gt_trajectory, current_state], 
        autograd_vectorize=vectorize, 
        name="goal_reaching"
    )
    objective.add(goal_cost)
    
    # ========== TRAJECTORY FOLLOWING COSTS (seguir GT) ==========
    
    # Penaliza desviación de la trayectoria ground truth
    traj_follow_cost = th.AutoDiffCostFunction(
        [control_variables], 
        trajectory_following, 
        12,  # 12 valores: 6 errores en x + 6 errores en y (muestreados cada 2 timesteps)
        cost_function_weights[7], 
        aux_vars=[gt_trajectory, current_state], 
        autograd_vectorize=vectorize, 
        name="trajectory_following"
    )
    objective.add(traj_follow_cost)
    
    # Peso 8: usar para goal_reaching también (para evitar warning)
    goal_cost_8 = th.AutoDiffCostFunction(
        [control_variables], 
        goal_reaching, 
        2,
        cost_function_weights[8],  # Asocia weight[8]
        aux_vars=[gt_trajectory, current_state], 
        autograd_vectorize=vectorize, 
        name="goal_reaching_aux"
    )
    objective.add(goal_cost_8)
    
    # ========== LEGACY COSTS (comentados, dependen del mapa) ==========
    
    # travel efficiency (depende del mapa para límites de velocidad)
    # speed_cost = th.AutoDiffCostFunction([control_variables], speed, trajectory_len, cost_function_weights[0], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="speed")
    # objective.add(speed_cost)
    
    # lane following (depende del mapa para carriles)
    # lane_xy_cost = th.AutoDiffCostFunction([control_variables], lane_xy, trajectory_len, cost_function_weights[5], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_xy")
    # objective.add(lane_xy_cost)
    # lane_theta_cost = th.AutoDiffCostFunction([control_variables], lane_theta, trajectory_len//2, cost_function_weights[6], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_theta")
    # objective.add(lane_theta_cost)

    # traffic rules (depende del mapa para semáforos)
    # red_light_cost = th.AutoDiffCostFunction([control_variables], red_light_violation, trajectory_len, cost_function_weights[7], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="red_light")
    # objective.add(red_light_cost)
    
    # safety with Frenet (depende del mapa para sistema de coordenadas Frenet)
    # safety_cost = th.AutoDiffCostFunction([control_variables], safety, 10, cost_function_weights[8], aux_vars=[predictions, current_state, ref_line], autograd_vectorize=vectorize, name="safety")
    # objective.add(safety_cost)

    return objective
