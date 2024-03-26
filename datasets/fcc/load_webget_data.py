import numpy as np
import datetime
import tqdm
import os
import argparse

# Reference: Pensieve

FILE_PATH = './202302/curr_webget.csv'
OUTPUT_PATH = './cooked_202302/'
NUM_LINES = np.inf
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default=FILE_PATH)
parser.add_argument('--output_path', type=str, default=OUTPUT_PATH)
parser.add_argument('--num_lines', type=int, default=NUM_LINES)
args = parser.parse_args()

FILE_PATH = args.file_path
OUTPUT_PATH = args.output_path
NUM_LINES = args.num_lines

if not os.path.exists(OUTPUT_PATH):
	os.makedirs(OUTPUT_PATH)


bw_measurements = {}
def main():
	line_counter = 0
	with open(FILE_PATH, 'r') as f:
		for line in f:
			try:
				parse = line.split(',')
				uid = parse[0]
				dtime = (datetime.datetime.strptime(parse[1],'%Y-%m-%d %H:%M:%S') 
					- TIME_ORIGIN).total_seconds()
				target = parse[2]
				address = parse[3]
				throughput = parse[6]  # bytes per second
			except:
				continue

			k = (uid, target)
			if k in bw_measurements:
				bw_measurements[k].append(throughput)
			else:
				bw_measurements[k] = [throughput]

			line_counter += 1
			if line_counter >= NUM_LINES:
				break
			if line_counter % 500 == 0:
				print('Processed', line_counter, 'lines', end = '\r')

	for k in bw_measurements:
		out_file = 'trace_' + '_'.join(k)
		out_file = out_file.replace(':', '-')
		out_file = out_file.replace('/', '-')
		out_file = OUTPUT_PATH + out_file

		with open(out_file, 'w') as f:
			for i in bw_measurements[k]:
				f.write(str(i) + '\n')

if __name__ == '__main__':
	main()

