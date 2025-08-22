# used to combine the multiple instances of the training files if needed
import os


ROUGH_DATA_DIR = r"data/rough_data"
NUM_ROUGH_DATA = 5
REPLAY_BUFFER_SIZE = 1000000
POS_PER_FILE = []
for idx in range(NUM_ROUGH_DATA):
    POS_PER_FILE.append(0)

GAME_OUTCOMES_NAME = "_Selfgen EosZero Run Game Outcomes"
MOVE_VISITS_NAME = "_Selfgen EosZero Run Move Visits"
POSITIONS_VECTORS_NAME = "_Selfgen EosZero Run Positions Vectors"

THIS_GEN_DATA_DIR = os.path.join(ROUGH_DATA_DIR, "this_gen_data_res")
os.makedirs(THIS_GEN_DATA_DIR, exist_ok=True)

LAST_GEN_DATA_DIR = os.path.join(ROUGH_DATA_DIR, "last_gen_data")
os.makedirs(LAST_GEN_DATA_DIR, exist_ok=True)

LAST_GEN_2_DATA_DIR = os.path.join(ROUGH_DATA_DIR, "last_gen_2_data")
os.makedirs(LAST_GEN_2_DATA_DIR, exist_ok=True)

COMBINED_DATA_RES = os.path.join(ROUGH_DATA_DIR, "combined_data_res")
os.makedirs(COMBINED_DATA_RES, exist_ok=True)

def generate_table_row(text, length: int):
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


def concat_files(file_name_type:str, num_rough_files:int):
    write_file_temp = []
    for i in range(num_rough_files):
        lines_in_file = 0
        if i > 0:
            write_file_temp.append("\n")

        with open(f"{ROUGH_DATA_DIR}/{file_name_type}{i+1}.txt") as rough_data:
            for line in rough_data:
                write_file_temp.append(line)
                lines_in_file += 1

        POS_PER_FILE[i] = lines_in_file

    with open(f"{THIS_GEN_DATA_DIR}/{file_name_type}.txt", "w") as f:
        f.writelines(write_file_temp)


def gen_combine_names(general_name:str, num_files:int, input_path:str):
    combine_names = []

    for i in range(num_files):
        combine_names.append(f"{input_path}/{general_name}{i + 1}")

    return combine_names


def combine_files(paths:list[str], output_dir:str, output_name:str, max_len:int = 500000):
    write_file_temp = []
    lines_in_each_file = []

    for i in range(len(paths)):
        lines_in_this_file = 0
        lines_in_each_file.append(0)

        if len(write_file_temp) >= max_len:
            break

        if i > 0:
            write_file_temp[-1] = f"{write_file_temp[-1]}\n"

        with open(f"{paths[i]}.txt") as rough_data:
            for line in rough_data:

                if len(write_file_temp) >= max_len:
                    write_file_temp[-1] = write_file_temp [-1].strip('\n')
                    break

                write_file_temp.append(line)
                lines_in_this_file += 1

        lines_in_each_file[i] = lines_in_this_file

    # now let's write all the combined entries into one file
    with open(f"{output_dir}/{output_name}.txt", "w") as f:
        f.writelines(write_file_temp)

    return lines_in_each_file


print("Starting file assembly...")


# now let's use the function to concatenate all the various datapoints
combine_this_gen_files_len = [
    combine_files(gen_combine_names(GAME_OUTCOMES_NAME, NUM_ROUGH_DATA, ROUGH_DATA_DIR), THIS_GEN_DATA_DIR,
                  GAME_OUTCOMES_NAME),
    combine_files(gen_combine_names(MOVE_VISITS_NAME, NUM_ROUGH_DATA, ROUGH_DATA_DIR), THIS_GEN_DATA_DIR,
                  MOVE_VISITS_NAME),
    combine_files(gen_combine_names(POSITIONS_VECTORS_NAME, NUM_ROUGH_DATA, ROUGH_DATA_DIR), THIS_GEN_DATA_DIR,
                  POSITIONS_VECTORS_NAME)]


print(f"\nAll Files Lengths: {combine_this_gen_files_len}")

print("\nFile assembly (pt 1.) successfully completed!\n")


#give the user more info on how many positions are in each file
total_positions = 0
for idx in range(len(combine_this_gen_files_len[0])):
    print(f"File Batch {idx + 1}: {combine_this_gen_files_len[0][idx]} positions")
    total_positions += combine_this_gen_files_len[0][idx]

print(f"\nTotal Positions (all files in this generation): {total_positions}")

print("\nFile assembly (pt 2. - replay buffer) started...\n")
# part 2 will combine current gen's data, along with the last 2 gens data, with a max length of 500k entries
# we will load this gen's data first, to ensure they all get loaded in fine, followed by last gen's data, then finally the one from 2 gens ago
# this effectively acts as a "priority" system

def gen_replay_buffer_paths(name):
    replay_buffer_paths = [
        f"{THIS_GEN_DATA_DIR}/{name}",
        f"{LAST_GEN_DATA_DIR}/{name}",
        f"{LAST_GEN_2_DATA_DIR}/{name}"
    ]

    return replay_buffer_paths


# let's now make the replay buffer by combining this gen's data with last gen's
combine_replay_buffer = [
    combine_files(gen_replay_buffer_paths(GAME_OUTCOMES_NAME), COMBINED_DATA_RES, GAME_OUTCOMES_NAME, REPLAY_BUFFER_SIZE),
    combine_files(gen_replay_buffer_paths(POSITIONS_VECTORS_NAME), COMBINED_DATA_RES, POSITIONS_VECTORS_NAME, REPLAY_BUFFER_SIZE),
    combine_files(gen_replay_buffer_paths(MOVE_VISITS_NAME), COMBINED_DATA_RES, MOVE_VISITS_NAME, REPLAY_BUFFER_SIZE)
]

replay_buffer_total_size = 0
for idx in range(len(combine_replay_buffer[0])):
    replay_buffer_total_size += combine_replay_buffer[0][idx]

print("File assembly (pt 2. - replay buffer) successfully completed!\n")

generate_table_row(["-", "-", "-"], 20)
generate_table_row(["Replay Buffer Data", " ", " "], 20)
generate_table_row(["-", "-", "-"], 20)
generate_table_row(["Type", "Size", "Percentage"], 20)
generate_table_row(["-", "-", "-"], 20)
generate_table_row(["Current Gen", combine_replay_buffer[0][0], f"{round(100 * combine_replay_buffer[0][0] / replay_buffer_total_size, 3)}%"], 20)
generate_table_row(["Last Gen", combine_replay_buffer[0][1], f"{round(100 * combine_replay_buffer[0][1] / replay_buffer_total_size, 3)}%"], 20)
generate_table_row(["Last 2 Gen", combine_replay_buffer[0][2], f"{round(100 * combine_replay_buffer[0][2] / replay_buffer_total_size, 3)}%"], 20)
generate_table_row(["-", "-", "-"], 20)
generate_table_row(["Total", replay_buffer_total_size, f"{round(100 * replay_buffer_total_size / replay_buffer_total_size, 3)}%"], 20)
generate_table_row(["-", "-", "-"], 20)