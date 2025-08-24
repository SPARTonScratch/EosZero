import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import trange
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
import math
import json
import torch.nn.functional as F

matplotlib.use('Agg')


def generate_table_row(text, length: int):
    """
    Generates a single row for a console-based table.
    """
    print_line = '|'
    for val in text:
        val = str(val)
        if val == "-":
            val = '-' * (length + 1)
            current_row = val + "|"
        else:
            current_row = " " + val
            current_row += " " * (length - len(current_row))
            current_row += " |"

        print_line += current_row
    print(print_line)


# --- Model and Data Loading Definitions ---
def load_2hot_states(path):
    """
    Load Connect 4 game states from a text file with 2-hot encoding
    Returns: numpy array of shape (num_samples, 85)
    """
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue

            bits = []
            for double in parts[:-1]:
                a, b = double.split(',')
                bits += [int(a), int(b)]

            stm = int(parts[-1])
            rows.append(bits + [stm])

    return np.array(rows, dtype=np.float32)


class EosZeroNet(nn.Module):
    """
    Neural Network architecture for Connect 4 policy and value prediction.
    """
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(85, 128),
            nn.ReLU(),
        )

        # Policy-specific processing branch
        self.policy_intermediate = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(16, 7)

        # Value-specific processing branch
        self.value_intermediate = nn.Sequential(
            nn.Linear(128, 8),
            nn.ReLU()
        )
        self.value_head = nn.Linear(8, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for the shared layers
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0.1)

        # Initialize weights for the policy branch
        for layer in self.policy_intermediate:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0.1)
        init.normal_(self.policy_head.weight, std=0.01)
        init.constant_(self.policy_head.bias, 0.0)

        # Initialize weights for the value branch
        for layer in self.value_intermediate:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.constant_(layer.bias, 0.1)
        init.normal_(self.value_head.weight, std=0.01)
        init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        # Pass input through the shared layers
        shared_features = self.shared(x)

        policy_features = self.policy_intermediate(shared_features)
        value_features = self.value_intermediate(shared_features)

        # Final prediction heads
        policy = self.policy_head(policy_features)
        value = self.value_head(value_features)

        return policy, value


def clean_up_export_data(input_data):
    """
    Cleans up the weights and biases to make them a single array.
    """
    cleaned_export_data = input_data.split(",")

    idx = 0
    for val in cleaned_export_data:
        cleaned_export_data[idx] = float((val.replace('[', '')).replace(']', ''))
        idx += 1

    return cleaned_export_data


# --- Export Functionality ---
def export_data(model, export_name, export_dir):
    """
    Loads a PyTorch model and exports its weights and biases to JSON and text files.
    """
    print(f"\nExtracting and exporting weights for '{export_name}'...")
    model.eval()

    export_out_dir = os.path.join("net_checkpoints", export_dir, export_name, "uncleaned")
    os.makedirs(export_out_dir, exist_ok=True)

    export_out_dir_clean = os.path.join("net_checkpoints", export_dir, export_name, "cleaned")
    os.makedirs(export_out_dir_clean, exist_ok=True)

    scratch_weights = {}

    for name, param in model.named_parameters():
        key = name.replace('.', '_')
        data = param.detach().cpu().tolist()
        scratch_weights[key] = data

    print(f"Export started...")
    combined_path = os.path.join(export_out_dir, "all_scratch_weights.json")
    with open(combined_path, "w") as f:
        json.dump(scratch_weights, f, indent=2)

    for key, data in scratch_weights.items():
        txt_path = os.path.join(export_out_dir, f"{key}.txt")
        with open(txt_path, "w") as f:
            f.write(json.dumps(data))

        txt_path = os.path.join(export_out_dir_clean, f"{key} (cleaned).txt")
        with open(txt_path, "w") as f:
            cleaned_up = clean_up_export_data(json.dumps(data))
            length = len(cleaned_up)
            for index, item in enumerate(cleaned_up):
                f.write(f"{item}")
                if index < (length - 1):
                    f.write(f"\n")

    print(f"Export complete! 🎉")
    print(f"- Combined JSON:  {combined_path}")
    print(f"- Individual .txt files in: {export_out_dir}/")


# --- Main Training Function ---
def train(states_file, visits_file, outcomes_file, net_name="EosZeroNet", epochs=100, batch_size=128, val_split=0.2):
    """
    Main training function with added checkpointing and export.
    Returns: a list of dictionaries containing metrics for each epoch.
    """
    # ======================
    # 1. Setup and Config
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    use_threads = math.ceil((os.cpu_count() / 2))
    torch.set_num_threads(use_threads)
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(CHECKPOINTS_BASE_DIR, exist_ok=True)
    if str(device) == "cpu":
        print(f"\nStarting Training on {use_threads} CPU Threads")
    else:
        print(f"\nStarting Training on GPU")

    print(f"Checkpoints will be saved to: {CHECKPOINTS_BASE_DIR}/{NET_NAME}-[epoch]/")

    # ======================
    # 2. Data Preparation
    # ======================
    print("\nLoading data...")
    data_load_time_start = time.time()
    visit_counts = np.loadtxt(visits_file, delimiter=",")
    visit_counts = np.clip(visit_counts, 1e-5, None)
    visit_counts = visit_counts / visit_counts.sum(axis=1, keepdims=True)

    print("\nProcessed Dataset Visit Counts")

    dataset = TensorDataset(
        torch.as_tensor(load_2hot_states(states_file), dtype=torch.float32),
        torch.as_tensor(visit_counts, dtype=torch.float32),
        torch.as_tensor(np.loadtxt(outcomes_file), dtype=torch.float32)
    )
    print("\nCreated Dataset")

    # ======================
    # 3. Data Split & Loaders
    # ======================
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    data_load_time_end = time.time()
    print(f"\nFinal Dataset Split: ({round(data_load_time_end - data_load_time_start, 3)}s)")
    generate_table_row(["-", "-", "-"], 15)
    generate_table_row(["Set", "Samples", "Percentage"], 15)
    generate_table_row(["-", "-", "-"], 15)
    generate_table_row(["Training", train_size, f"{100 * (1 - val_split)}%"], 15)
    generate_table_row(["Validation", val_size, f"{100 * val_split}%"], 15)
    generate_table_row(["-", "-", "-"], 15)
    generate_table_row(["Total", len(dataset), "100.0%"], 15)
    generate_table_row(["-", "-", "-"], 15)

    train_workers = math.ceil(os.cpu_count() / 1.4)
    val_workers = math.ceil((train_workers / 3.5) + 0.1)
    print(f"\nTrainer Loader Workers: {train_workers}")
    print(f"Validation Loader Workers: {val_workers}")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, num_workers=val_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    # ======================
    # 4. Model Setup
    # ======================
    model = EosZeroNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # ======================
    # 5. Training Setup
    # ======================

    training_configs = [
        {"init_lr": 0.015, "scheduler": "cosine_annealing", "eta_min_lr": 1e-5, "cycle": epochs},
        {"init_lr": 0.008, "scheduler": "step_lr", "step_size": 125, "gamma": 0.5},
        {"init_lr": 0.015, "scheduler": "cosine_annealing_restart", "eta_min_lr": 2e-5, "cycle": math.ceil(epochs / 6.25), "t_mult": 2},
        {"init_lr": 0.010, "scheduler": "cosine_annealing_restart", "eta_min_lr": 5e-6, "cycle": math.floor(epochs / 7), "t_mult": 2}
    ]

    use_training_config = 3

    init_lr = training_configs[use_training_config].get("init_lr")
    scheduler_type = training_configs[use_training_config].get("scheduler")
    eta_min_lr = training_configs[use_training_config].get("eta_min_lr")
    step_size = training_configs[use_training_config].get("step_size")
    step_lr_gamma = training_configs[use_training_config].get("gamma")
    cycle_len = training_configs[use_training_config].get("cycle")
    if cycle_len == 0:
        cycle_len = epochs
        print(f"Cycle len set to: {cycle_len} due to invalid val")
    cycle_mult = training_configs[use_training_config].get("t_mult")

    print(f"Training Configs: {training_configs[use_training_config]}")

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=2e-5, betas=(0.9, 0.999))

    if scheduler_type == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_lr_gamma)

    elif scheduler_type == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=cycle_len, eta_min=eta_min_lr)

    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cycle_len,
            T_mult=cycle_mult,
            eta_min=eta_min_lr
        )

    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    # ======================
    # 6. Training Loop with Checkpoints
    # ======================
    print("\n🔥 Starting training... (Ctrl+C to save and exit early)")
    all_metrics = []
    train_losses = []
    val_losses = []
    val_policy_accuracies = []
    val_value_maes = [] # mean absolute errors
    lr_values = []

    timing_metrics = []

    value_loss_strength = 0.9 # Total_loss = policy_loss_val + x * value_loss_val, 0.1 to 1 seems like a good range

    try:
        with trange(epochs, desc="Epochs") as pbar:
            for epoch in pbar:
                # Timer for the entire epoch
                epoch_start_time = time.time()

                # --- Training phase ---
                model.train()
                epoch_train_loss = 0.0
                epoch_train_loss_policy = 0.0
                epoch_train_loss_value = 0.0

                total_training_step_time = 0.0

                train_start_time = time.time()
                for states, policies, values in train_loader:
                    states, policies, values = states.to(device, non_blocking=True), policies.to(device, non_blocking=True), values.to(device, non_blocking=True).unsqueeze(1)

                    # Timer for the training step (forward, backward, optimize)
                    training_step_start_time = time.time()
                    optimizer.zero_grad()
                    pred_p, pred_v = model(states)

                    log_probs = F.log_softmax(pred_p, dim=1)
                    policy_loss_val = policy_loss_fn(log_probs, policies)
                    epoch_train_loss_policy += policy_loss_val.item()

                    pred_v = torch.tanh(pred_v)
                    value_loss_val = value_loss_fn(pred_v, values)
                    epoch_train_loss_value += value_loss_val.item()

                    total_loss = policy_loss_val + value_loss_strength * value_loss_val
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Now that the core "training part" (forward, backward, optimize) is done, let's log it
                    training_step_end_time = time.time()
                    total_training_step_time += (training_step_end_time - training_step_start_time)

                    epoch_train_loss += total_loss.item()

                # Data loading time is implicitly measured by the time taken between steps.
                # In other words, it's the difference between total training time and total step time.
                train_end_time = time.time()
                total_train_time = train_end_time - train_start_time
                total_data_load_time = total_train_time - total_training_step_time

                # --- Validation phase ---
                model.eval()
                epoch_val_loss = 0.0
                epoch_val_loss_policy = 0.0
                epoch_val_loss_value = 0.0

                epoch_val_policy_accuracy = 0.0
                epoch_val_value_mae = 0.0 # mean absolute error
                val_start_time = time.time()
                with torch.no_grad():
                    for states, policies, values in val_loader:
                        states, policies, values = states.to(device, non_blocking=True), policies.to(device, non_blocking=True), values.to(device, non_blocking=True).unsqueeze(1)
                        pred_p, pred_v = model(states)

                        log_probs = F.log_softmax(pred_p, dim=1)
                        policy_loss_val = policy_loss_fn(log_probs, policies)
                        epoch_val_loss_policy += policy_loss_val.item()

                        pred_v = torch.tanh(pred_v)
                        value_loss_val = value_loss_fn(pred_v, values)
                        epoch_val_loss_value += value_loss_val.item()

                        total_loss = policy_loss_val + value_loss_strength * value_loss_val
                        epoch_val_loss += total_loss.item()

                        predicted_moves = torch.argmax(pred_p, dim=1)
                        ground_truth_moves = torch.argmax(policies, dim=1)
                        correct_move_predictions = (predicted_moves == ground_truth_moves).sum().item()
                        epoch_val_policy_accuracy += correct_move_predictions

                        values_mae_val = abs(pred_v - values).mean()
                        epoch_val_value_mae += values_mae_val.item()

                avg_val_policy_accuracy = epoch_val_policy_accuracy / len(val_set)
                avg_val_value_mae = epoch_val_value_mae / len(val_loader)

                val_end_time = time.time()
                total_val_time = val_end_time - val_start_time

                current_lr = optimizer.param_groups[0]['lr'] # let's set the logging lr before the scheduler changes it

                scheduler.step()

                # --- End of epoch timing ---
                epoch_end_time = time.time()
                total_epoch_time = epoch_end_time - epoch_start_time

                def get_avg_loss_train(val):
                    return val / len(train_loader)

                def get_avg_loss_val(val):
                    return val / len(val_loader)

                avg_train_loss = get_avg_loss_train(epoch_train_loss)
                avg_val_loss = get_avg_loss_val(epoch_val_loss)

                avg_train_loss_value = get_avg_loss_train(epoch_train_loss_value)
                avg_train_loss_policy = get_avg_loss_train(epoch_train_loss_policy)

                avg_val_loss_value = get_avg_loss_val(epoch_val_loss_value)
                avg_val_loss_policy = get_avg_loss_val(epoch_val_loss_value)

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                val_policy_accuracies.append(avg_val_policy_accuracy * 100) # multiply by 100 to convert to percentage
                val_value_maes.append(avg_val_value_mae)
                lr_values.append(current_lr)

                checkpoint_dir = os.path.join(CHECKPOINTS_BASE_DIR, f"{net_name}_epoch_{epoch + 1}")
                checkpoint_path = os.path.join(checkpoint_dir, f"{net_name}_epoch_{epoch + 1}.pth")

                best_net_dir = BEST_NET_DIR
                best_net_path = BEST_NET_PATH

                # --- Store all metrics ---
                all_metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_loss_value": avg_train_loss_value,
                    "train_loss_policy": avg_train_loss_policy,
                    "val_loss_value": avg_val_loss_value,
                    "val_loss_policy": avg_val_loss_policy,
                    "val_policy_accuracy": avg_val_policy_accuracy,
                    "val_value_mae": avg_val_value_mae,
                    "lr": current_lr,
                    "model_path": checkpoint_path
                })

                # --- Add profiling metrics for this epoch ---
                timing_metrics.append({
                    "epoch": epoch + 1,
                    "total_epoch_time_s": total_epoch_time,
                    "total_train_time_s": total_train_time,
                    "total_val_time_s": total_val_time,
                    "avg_data_load_time_ms": (total_data_load_time / len(train_loader)) * 1000,
                    "total_data_load_time_s": total_data_load_time,
                    "avg_training_step_time_ms": (total_training_step_time / len(train_loader)) * 1000,
                    "total_step_time_s": total_training_step_time,
                    "num_batches": len(train_loader)
                })

                pbar.set_postfix({
                    'train_loss': f"{avg_train_loss:.4f}",
                    'val_loss': f"{avg_val_loss:.4f}",
                    'gap': f"{avg_train_loss - avg_val_loss:+.4f}",
                    'lr': f"{current_lr:.4e}"
                })

                global lowest_val_loss

                if avg_val_loss <= lowest_val_loss:
                    # is the "best net" according to validation loss
                    global lowest_val_loss_epoch

                    lowest_val_loss = avg_val_loss
                    lowest_val_loss_epoch = epoch + 1

                    os.makedirs(best_net_dir, exist_ok=True)
                    torch.save(model.state_dict(), best_net_path)

                # Create a new directory for this epoch's checkpoint if the net should be saved now
                if epoch + 1 == epochs or (epoch + 1) % SAVE_RATE == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted! (However, no checkpoints will be lost)")

    # --- Plotting functionality ---
    # Create a list of epoch numbers from 1 to the total number of epochs
    epochs_range = range(1, epochs + 1)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        # Pass epochs_range as the x-axis values
        plt.plot(epochs_range, train_losses, label='Training', color='royalblue')
        plt.plot(epochs_range, val_losses, label='Validation', color='darkorange')

        ax = plt.gca()

        ax.xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(epochs / 20)))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.010))

        plt.title(f"Training vs Validation Loss ({net_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{BEST_NET_DIR}/training_curves_{net_name}.png")
        print(f"\nLoss curves saved to training_curves_{net_name}.png")

        plt.figure(figsize=(15, 10))
        # Pass epochs_range as the x-axis values
        plt.plot(epochs_range, val_policy_accuracies, label='Policy Accuracy', color='darkorange')
        plt.title(f"Validation Policy Accuracy ({net_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{BEST_NET_DIR}/policy_accuracy_{net_name}.png")
        print(f"Policy accuracy curve saved to policy_accuracy_{net_name}.png")

        plt.figure(figsize=(15, 10))
        # Pass epochs_range as the x-axis values
        plt.plot(epochs_range, val_value_maes, label='Value Prediction Error (MAE)', color='darkorange')
        plt.title(f"Validation Value Prediction Error ({net_name})")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{BEST_NET_DIR}/value_prediction_error_{net_name}.png")
        print(f"Value error curve saved to value_prediction_error_{net_name}.png")

        plt.figure(figsize=(15, 10))
        # Pass epochs_range as the x-axis values
        plt.plot(epochs_range, lr_values, label='Training Learn Rate', color='royalblue')
        plt.title(f"Training Learn Rate ({net_name})")
        plt.xlabel("Epochs")
        plt.ylabel("Learn Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{BEST_NET_DIR}/training_learn_rate_{net_name}.png")
        print(f"Learn rate curve saved to training_learn_rate_{net_name}.png")

    except ImportError:
        print("\nMatplotlib not found. Cannot save loss curves.")

    # --- Export timing metrics to a JSON file ---
    timing_path = os.path.join(BEST_NET_DIR, "timing_metrics.json")
    with open(timing_path, "w") as f:
        json.dump(timing_metrics, f, indent=2)
    print(f"Timing metrics saved to {timing_path}")

    training_settings_dump = [
        f"Net Name: {net_name}",
        f"Training Configs: {training_configs[use_training_config]}"
    ]

    # --- Export training settings to a JSON file ---
    training_settings_path = os.path.join(BEST_NET_DIR, "training_settings.json")
    with open(training_settings_path, "w") as f:
        json.dump(training_settings_dump, f, indent=2)
    print(f"Training settings saved to {training_settings_path}")

    # --- Export all metrics to a JSON file ---
    all_metrics_path = os.path.join(BEST_NET_DIR, "all_metrics.json")
    with open(all_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"All metrics saved to {all_metrics_path}")

    return all_metrics


if __name__ == "__main__":
    # --- User-definable parameters ---
    NET_NAME = "PVN 1.32"
    EPOCHS = 1000
    BATCH_SIZE = 1024
    VAL_SPLIT = 0.2
    SAVE_RATE = 10
    CHECKPOINTS_BASE_DIR = "net_checkpoints"  # Base directory for all checkpoints
    BEST_NET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, f"{NET_NAME}_final_output")
    os.makedirs(BEST_NET_DIR, exist_ok=True)
    BEST_NET_PATH = os.path.join(BEST_NET_DIR, f"{NET_NAME}_best_net.pth")

    # --- helper vars ---
    lowest_val_loss = 999999
    lowest_val_loss_epoch = -1
    best_net_name = f"{NET_NAME}_best_net"

    # --- File paths ---
    STATES_FILE = "data/_Selfgen EosZero Run Positions Vectors.txt"
    VISITS_FILE = "data/_Selfgen EosZero Run Move Visits.txt"
    OUTCOMES_FILE = "data/_Selfgen EosZero Run Game Outcomes.txt"

    trained_metrics = train(STATES_FILE, VISITS_FILE, OUTCOMES_FILE,
                        net_name=NET_NAME,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        val_split=VAL_SPLIT)

    if trained_metrics:
        best_metric = trained_metrics[lowest_val_loss_epoch - 1] # since lists are zero-indexed, we must subtract 1 from epoch

        best_model_table_width = max(50, len(BEST_NET_PATH) + 5)
        generate_table_row(["-", "-"], best_model_table_width)
        generate_table_row(["Best Model Parameter", "Value"], best_model_table_width)
        generate_table_row(["-", "-"], best_model_table_width)
        generate_table_row(["Epoch:", str(best_metric['epoch'])], best_model_table_width)
        generate_table_row(["Train Loss:", str(round(best_metric['train_loss'], 4))], best_model_table_width)
        generate_table_row(["Validation Loss:", str(round(best_metric['val_loss'], 4))], best_model_table_width)
        generate_table_row(["-", "-"], best_model_table_width)
        generate_table_row(["Best Model Path:", BEST_NET_PATH], best_model_table_width)
        generate_table_row(["-", "-"], best_model_table_width)

        best_model_path = BEST_NET_PATH
        best_model = EosZeroNet()
        best_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        export_data(best_model, f"{NET_NAME}_best_epoch_{best_metric['epoch']}", export_dir=f"{NET_NAME}_final_output")

    else:
        print("No models were trained or saved.")