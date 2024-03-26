import numpy as np
import datetime
import tqdm
import os
import argparse
import random


# Reference: Pensieve
months = ['01', '02', '04', '05', '06']
for month in months:

    FILE_PATH = f'./2023{month}_cooked/'
    OUTPUT_FILE = f'./2023{month}_cooked_cleaned.txt'
    NUM_LINES = np.inf
    TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)

    file_list = []
    files = 0
    discard = 0
    for file_name in os.listdir(FILE_PATH):
        file_path = os.path.abspath(os.path.join(FILE_PATH, file_name))
        with open(file_path, 'r') as file:
            if files >= NUM_LINES: break
            lines = file.readlines()
            total_lines = sum(1 for l in lines)
            rand = random.randint(5, total_lines-6) if 5 < total_lines - 6 else None
            first_five_lines = [int(line.strip()) for line in lines[:5]]
            last_five_lines = [int(line.strip()) for line in lines[-5:]]
            mid_five_lines = [float(line.strip()) for line in file.readlines()[rand:rand+5]] if rand is not None else None
            #lines = first_five_lines + last_five_lines + mid_five_lines if mid_five_lines else first_five_lines + last_five_lines
            if all(line == 0 for line in first_five_lines) or all(line == 0 for line in last_five_lines) or mid_five_lines and all(line == 0 for line in mid_five_lines):
                discard += 1
            else:
                file_list.append(file_path)
                files += 1
    
    print("discarded", discard, "traces")

    with open(OUTPUT_FILE, 'w') as output_file:
        for file_name in file_list:
            output_file.write(file_name + '\n')