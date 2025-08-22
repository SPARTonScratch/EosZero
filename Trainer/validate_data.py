import numpy as np
import argparse

def validate_data(states_path, visits_path, outcomes_path):
    """Comprehensive data validation function for 2-hot encoded data"""
    print(f"Validating states from: {states_path}")
    states = []
    data_stm = []
    with open(states_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            # 42 cells (2 bits each) + 1 STM bit = 85
            if len(parts) != 43:
                raise ValueError(f"Invalid line {i + 1}: Expected 43 parts, got {len(parts)}. Line: {line[:50]}...")

            vec = []
            for double in parts[:-1]:
                if double.count(',') != 1:
                    raise ValueError(f"Invalid double '{double}' on line {i + 1}. Expected one comma.")
                a, b = map(int, double.split(','))
                if sum([a, b]) > 1:
                    raise ValueError(f"Invalid 2-hot encoding on line {i + 1}. Sum of bits must be 0 or 1.")
                vec.extend([a, b])

            stm = int(parts[-1])
            if stm not in {0, 1}:
                raise ValueError(f"Invalid STM bit '{stm}' on line {i + 1}. Must be 0 or 1.")
            data_stm.append(stm)
            vec.append(stm)
            states.append(vec)

    # Check the total size of the state vector
    if len(states[0]) != 85:
        raise ValueError(f"Invalid state vector length. Expected 85, got {len(states[0])}")

    data_stm_np = np.array(data_stm)

    print("✅ States validation passed.")

    print(f"\nValidating visits from: {visits_path}")
    visits = np.loadtxt(visits_path, delimiter=',')
    if visits.min() < 0 or visits.max() > 1:
        raise ValueError("Visit probabilities outside [0,1] range.")

    sums = visits.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        bad_indices = np.where(~np.isclose(sums, 1.0, atol=1e-4))[0]
        raise ValueError(f"Invalid visit sums at rows: {bad_indices[:10]}... Sums must equal 1.0.")

    print("✅ Visits validation passed.")

    print(f"\nValidating outcomes from: {outcomes_path}")
    outcomes = np.loadtxt(outcomes_path)
    if not set(outcomes).issubset({-1, 0, 1}):
        invalid = set(outcomes) - {-1, 0, 1}
        raise ValueError(f"Invalid outcome values: {invalid}. Outcomes must be -1, 0, or 1.")

    print("✅ Outcomes validation passed.")

    # now it's time to combine the outcomes, which are side-to-move (or stm relative), to the STMs for each pos

    num_draw_mask = (outcomes == 0)
    num_draws = np.sum(num_draw_mask)

    num_red_win_mask = ((outcomes == 1) & (data_stm_np == 1)) | ((outcomes == -1) & (data_stm_np == 0))
    num_red_wins = np.sum(num_red_win_mask)

    num_yellow_win_mask = (outcomes == 1) & (data_stm_np == 0) | ((outcomes == -1) & (data_stm_np == 1))
    num_yellow_wins = np.sum(num_yellow_win_mask)

    num_total_positions = num_yellow_wins + num_red_wins + num_draws


    def get_display(num_res):
        return f"{(num_res / num_total_positions) * 100:.3f}%"


    red_win_display = get_display(num_red_wins)
    draw_display = get_display(num_draws)
    yellow_win_display = get_display(num_yellow_wins)

    print("\nFinal Result:")

    if len(outcomes) == num_total_positions:
        print("✅ All data validation checks passed!")
    else:
        print(f"W+D+L outcomes ({num_total_positions}) are not equal to num outcomes ({len(outcomes)})")

    print("\n")
    print(f"Total Draws: {num_draws}")
    print(f"Total Red Wins: {num_red_wins}")
    print(f"Total Yellow Wins: {num_yellow_wins}")
    print(f"Total Positions: {num_total_positions}")
    print(f"Red: {red_win_display} - Draw: {draw_display} - Yellow: {yellow_win_display}")

    def print_colored_bar(char, length, color_code):
        bar = char * length
        return f"\033[{color_code}m{bar}\033[0m"

    distribution_bar_len = 50

    red_distribution_bar_num = round((num_red_wins / num_total_positions) * distribution_bar_len)
    yellow_distribution_bar_num = round((num_yellow_wins / num_total_positions) * distribution_bar_len)
    draw_distribution_bar_num = distribution_bar_len - (red_distribution_bar_num + yellow_distribution_bar_num)

    red_bar_display = print_colored_bar("█", red_distribution_bar_num, 91)
    draw_bar_display = print_colored_bar("█", draw_distribution_bar_num, 90)
    yellow_bar_display = print_colored_bar("█", yellow_distribution_bar_num, 93)

    print(f"Distribution: {red_bar_display}{draw_bar_display}{yellow_bar_display}")

    return True


if __name__ == "__main__":
    # Default paths
    default_paths = {
        'states': "data/_Selfgen EosZero Run Positions Vectors.txt",
        'visits': "data/_Selfgen EosZero Run Move Visits.txt",
        'outcomes': "data/_Selfgen EosZero Run Game Outcomes.txt"
    }

    parser = argparse.ArgumentParser(description='Validate Training Data')
    parser.add_argument("--states", default=default_paths['states'],
                        help=f"Path to state vectors (default: {default_paths['states']})")
    parser.add_argument("--visits", default=default_paths['visits'],
                        help=f"Path to visit counts (default: {default_paths['visits']})")
    parser.add_argument("--outcomes", default=default_paths['outcomes'],
                        help=f"Path to outcomes (default: {default_paths['outcomes']})")

    args = parser.parse_args()

    print(f"Data Validation Started...\n")

    try:
        validate_data(args.states, args.visits, args.outcomes)
    except Exception as e:
        print(f"❌ Critical data issue: {e}")
        exit(1)