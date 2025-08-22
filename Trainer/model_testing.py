import torch
import torch.nn as nn
import numpy as np

# --- USER-EDITABLE CONFIGURATION ---
# Path to your trained PyTorch model file (.pth).
MODEL_PATH = "net_checkpoints/PVN 0.75_final_output/PVN 0.75_best_net.pth"

# Path to a text file containing one 2-hot encoded game state string per line.
# Example content of this file:
# "0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 0,0 1"

STATES_FILE_PATH = "data/model_testing_positions.txt"


# --- Model Definition (Must be an exact match to training code) ---
class EosZeroNet(nn.Module):
    """
    Neural Network architecture for Connect 4 policy and value prediction.
    """

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(85, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, 7)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        shared_features = self.shared(x)
        policy = torch.softmax(self.policy_head(shared_features), dim=1)
        value = torch.tanh(self.value_head(shared_features))
        return policy, value


def load_states_from_file(file_path):
    """
    Parses multiple 2-hot encoded state strings from a file into a list of PyTorch tensors.
    Each line in the file should represent a single game state.
    """
    states_tensors = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    parts = line.split()
                    if len(parts) != 43:
                        raise ValueError(
                            f"Line {i + 1}: Invalid state string length. Expected 43 parts, got {len(parts)}.")

                    bits = []
                    for double in parts[:-1]:
                        a, b = double.split(',')
                        bits += [int(a), int(b)]

                    stm = int(parts[-1])
                    bits.append(stm)

                    states_tensors.append(torch.as_tensor(bits, dtype=torch.float32).unsqueeze(0))
                except Exception as e:
                    print(f"Error parsing state on line {i + 1}: {e}")
                    continue  # Continue to next line even if one fails

        return states_tensors
    except FileNotFoundError:
        print(f"❌ Error: Input file '{file_path}' not found.")
        return None


if __name__ == "__main__":
    # --- 1. Load the model ---
    try:
        model = EosZeroNet()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        print(f"✅ Model loaded successfully from '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # --- 2. Prepare the input data ---
    states_to_eval = load_states_from_file(STATES_FILE_PATH)
    if states_to_eval is None or not states_to_eval:
        print(f"❌ No valid states were loaded from '{STATES_FILE_PATH}'.")
        exit(1)
    print(f"✅ Loaded {len(states_to_eval)} states from file.")
    print("--------------------------------------")

    # --- 3. Run a forward pass for each state ---
    print("--- Model Predictions for Each Position ---")
    with torch.no_grad():
        for i, input_tensor in enumerate(states_to_eval):
            policy, value = model(input_tensor)

            policy_list = policy.squeeze().tolist()
            value_list = value.squeeze().tolist()

            print(f"\nPosition {i + 1}:")
            print(f"Policy Output: {np.array2string(np.array(policy_list), precision=6, separator=', ')}")
            print(f"Value Output:  {value_list:.6f}")
    print("--------------------------------------")
