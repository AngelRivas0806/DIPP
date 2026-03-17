import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
import subprocess

from torch import nn, optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader

from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor, NUM_MODES

def train_epoch(data_loader, predictor, planner, optimizer, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)

        # ========================================================
        if batch_idx == 0:
            print(f"\n=== Tensor Shapes en GPU ===")
            print(f"ego.shape: {ego.shape}")
            print(f"neighbors.shape: {neighbors.shape}")
            print(f"ground_truth.shape: {ground_truth.shape}")
            print(f"ego device: {ego.device}")
        # ========================================================

        # current_state: (B, 11, feat) tomando el último frame observado
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

        # weights: máscara para vecinos presentes (no-0)
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

        # predict
        optimizer.zero_grad()
        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)

        if batch_idx == 0:
            weights_to_print = cost_function_weights[0].detach().cpu().numpy()
            np.set_printoptions(precision=4, suppress=True)
            print(f"\n{'='*50}")
            print(f"Inicio de la Época")
            print(f"Cost Weights: {weights_to_print}")
            print(f"{'='*50}\n")

        # convierto controles -> trayectorias (por modo)
        plan_trajs = torch.stack(
            [bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(NUM_MODES)],
            dim=1
        )

        # pérdida multi-futuro multi-agente
        # ego_in_pred_loss=False cuando use_planning: el ego se supervisa via imitation_loss sobre plan optimizado
        _, pred_loss, score_loss, best_mode = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights,
                                                         ego_in_pred_loss=not use_planning)

        # imitacion: ego del modo ganador vs gt — suma
        best_plan = plan_trajs[torch.arange(plan_trajs.shape[0], device=plan_trajs.device), best_mode]
        imitation_loss = F.smooth_l1_loss(best_plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

        # sin planning: loss base, cost_weights NO se usan aqui
        loss = args.lambda_pred * pred_loss + args.lambda_score * score_loss + args.lambda_imitation * imitation_loss

        # plan
        if use_planning:
            # selecciona el modo ganador para iniciar el optimizador
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(ego.shape[0], 24),
                "predictions": prediction,
                "current_state": current_state,
            }

            w = cost_function_weights  # (B, 5)

            planner_inputs["w_acc"]          = w[:, 0].unsqueeze(1)
            planner_inputs["w_jerk"]         = w[:, 1].unsqueeze(1)
            planner_inputs["w_steer"]        = w[:, 2].unsqueeze(1)
            planner_inputs["w_steer_change"] = w[:, 3].unsqueeze(1)
            planner_inputs["w_collision"]    = w[:, 4].unsqueeze(1)

            final_values, info = planner.layer.forward(planner_inputs)
            u_opt = final_values["control_variables"].view(-1, 12, 2)
            plan  = bicycle_model(u_opt, ego[:, -1])[:, :, :3]

            # Calcular cada residuo en PyTorch para que el gradiente llegue a cost_weights
            dt = 0.4
            w_acc          = w[:, 0].mean()
            w_jerk         = w[:, 1].mean()
            w_steer        = w[:, 2].mean()
            w_steer_change = w[:, 3].mean()
            w_col          = w[:, 4].mean()

            acc    = u_opt[:, :, 0]                            # (B, T)
            jerk   = torch.diff(acc,   dim=1) / dt            # (B, T-1)
            steer  = u_opt[:, :, 1]                           # (B, T)
            ds     = torch.diff(steer, dim=1) / dt            # (B, T-1)

            cost_reg = (w_acc   * acc.pow(2).mean()
                      + w_jerk  * jerk.pow(2).mean()
                      + w_steer * steer.pow(2).mean()
                      + w_steer_change * ds.pow(2).mean())

            # imitacion sobre plan optimizado — suma
            imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

            plan_cost = info.last_err.mean() if hasattr(info, "last_err") else 0

            loss = (args.lambda_pred * pred_loss
                    + args.lambda_score * score_loss
                    + args.lambda_imitation * imitation_loss
                    + args.lambda_cost * (cost_reg + w_col * plan_cost))
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        # metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # progress
        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rTrain Progress: [{current:>6d}/{size:>6d}]  "
            f"Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample"
        )
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]

    logging.info(
        f"\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, "
        f"predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}"
    )

    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        ground_truth = batch[2].to(args.device)

        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :2], 0)

        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors)
            plan_trajs = torch.stack(
                [bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(NUM_MODES)],
                dim=1
            )
            # ego_in_pred_loss=False cuando use_planning: el ego se supervisa via imitation_loss sobre plan optimizado
            _, pred_loss, score_loss, best_mode = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights,
                                                             ego_in_pred_loss=not use_planning)

            best_plan = plan_trajs[torch.arange(plan_trajs.shape[0], device=plan_trajs.device), best_mode]
            imitation_loss = F.smooth_l1_loss(best_plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

            loss = args.lambda_pred * pred_loss + args.lambda_score * score_loss + args.lambda_imitation * imitation_loss

        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(ego.shape[0], 24),
                "predictions": prediction,
                "current_state": current_state,
            }

            w = cost_function_weights  # (B, 5)

            planner_inputs["w_acc"]          = w[:, 0].unsqueeze(1)
            planner_inputs["w_jerk"]         = w[:, 1].unsqueeze(1)
            planner_inputs["w_steer"]        = w[:, 2].unsqueeze(1)
            planner_inputs["w_steer_change"] = w[:, 3].unsqueeze(1)
            planner_inputs["w_collision"]    = w[:, 4].unsqueeze(1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            u_opt = final_values["control_variables"].view(-1, 12, 2)
            plan  = bicycle_model(u_opt, ego[:, -1])[:, :, :3]

            plan_cost      = info.last_err.mean() if hasattr(info, "last_err") else 0
            imitation_loss = F.smooth_l1_loss(plan[:, :, :2], ground_truth[:, 0, :, :2], reduction='sum')

            loss = (args.lambda_pred * pred_loss
                    + args.lambda_score * score_loss
                    + args.lambda_imitation * imitation_loss
                    + args.lambda_cost * plan_cost)
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rValid Progress: [{current:>6d}/{size:>6d}]  "
            f"Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample"
        )
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]

    logging.info(
        f"\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, "
        f"val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}"
    )

    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, f"training_log/{args.name}/")
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=os.path.join(log_path, "train.log"))

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # predictor
    predictor = Predictor(12, predict_positions=args.predict_positions).to(args.device)

    # planner
    if args.use_planning:
        trajectory_len, feature_len = 12, 9
        planner = MotionPlanner(trajectory_len=trajectory_len, device=args.device)
    else:
        planner = None

    # optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.98)

    train_epochs = args.train_epochs
    batch_size = args.batch_size

    # dataset loaders
    if args.train_set.endswith(".npz"):
        train_set = DrivingData(args.train_set)
        valid_set = DrivingData(args.valid_set)
    else:
        train_set = DrivingData(args.train_set + "/*")
        valid_set = DrivingData(args.valid_set + "/*")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    # train loop
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")

        # si hay planner, pretrain predictor sin planning
        use_planning_now = args.use_planning
        if planner is not None:
            use_planning_now = (epoch >= args.pretrain_epochs)

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, use_planning_now)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, use_planning_now)

        log = {
            "epoch": epoch + 1,
            "loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val-loss": val_loss,
            "train-plannerADE": train_metrics[0],
            "train-plannerFDE": train_metrics[1],
            "train-predictorADE": train_metrics[2],
            "train-predictorFDE": train_metrics[3],
            "val-plannerADE": val_metrics[0],
            "val-plannerFDE": val_metrics[1],
            "val-predictorADE": val_metrics[2],
            "val-predictorFDE": val_metrics[3],
        }

        csv_path = os.path.join(script_dir, f"training_log/{args.name}/train_log.csv")
        if epoch == 0:
            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(csv_path, "a") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        scheduler.step()

        # save model
        torch.save(
            predictor.state_dict(),
            os.path.join(script_dir, f"training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth")
        )
        logging.info(f"Model saved in training_log/{args.name}\n")

    # plots
    logging.info("=" * 60)
    logging.info("Generating training graphs...")
    logging.info("=" * 60)

    try:
        log_csv_path = os.path.join(script_dir, f"training_log/{args.name}/train_log.csv")
        save_dir = os.path.join(script_dir, f"training_log/{args.name}")
        plot_script = os.path.join(script_dir, "plot_training.py")
        result = subprocess.run(
            ["python", plot_script, "--log_path", log_csv_path, "--save_dir", save_dir],
            capture_output=True,
            text=True,
            check=True,
            cwd=script_dir,
        )
        logging.info(f"Graphs saved in: training_log/{args.name}/")
        if result.stdout:
            logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error al generar gráficas: {e}")
        if e.stderr:
            logging.warning(e.stderr)
    except FileNotFoundError:
        logging.warning("No se encontró plot_training.py")

    logging.info("=" * 60)
    logging.info("ENTRENAMIENTO COMPLETADO")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--name", type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument("--train_set", type=str, help="path to train datasets", required=True)
    parser.add_argument("--valid_set", type=str, help="path to validation datasets", required=True)
    parser.add_argument("--seed", type=int, help="fix random seed", default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument("--pretrain_epochs", type=int, help="epochs of pretraining predictor", default=5)
    parser.add_argument("--train_epochs", type=int, help="epochs of training", default=20)
    parser.add_argument("--batch_size", type=int, help="batch size (default: 32)", default=32)
    parser.add_argument("--learning_rate", type=float, help="learning rate (default: 2e-4)", default=2e-4)
    parser.add_argument(
        "--use_planning",
        action="store_true",
        help="if use integrated planning module (default: False)",
        default=False,
    )
    parser.add_argument("--device", type=str, help="run on which device (default: cuda)", default="cuda")
    parser.add_argument("--lambda_pred",       type=float, default=0.7,  help="peso perdida de prediccion")
    parser.add_argument("--lambda_score",      type=float, default=1.0,  help="peso perdida de score")
    parser.add_argument("--lambda_imitation",  type=float, default=1.0,  help="peso perdida de imitacion")
    parser.add_argument("--lambda_cost",       type=float, default=1e-2, help="peso del costo Theseus (solo con planning)")
    parser.add_argument("--predict_positions", action="store_true", help="predecir posiciones (x,y) absolutas en vez de desplazamientos (dx,dy) para vecinos")
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        # fuerza el device global a 0 para que Theseus no intente cuda:1
        torch.cuda.set_device(0)
        # normaliza "cuda" -> "cuda:0"
        if args.device == "cuda":
            args.device = "cuda:0"

    model_training()