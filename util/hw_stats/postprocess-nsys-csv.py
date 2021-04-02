import csv
import argparse
import os

parser = argparse.ArgumentParser(description='Remove the memcpys from nsys csv')
parser.add_argument('--path', required=True, type=str,
                    help='path pointing where the nsys csv is stored')
args = parser.parse_args()

if __name__ == '__main__':
	
	compacted_csv = []
	cycles_location = os.path.join(args.path, 'cycles.csv')
	cycles_post_location = os.path.join(args.path, 'cycles_processed.csv')

	with open(cycles_location,'rw') as f:
		for line in f:
			if("memcpy" in line):
				continue
			compacted_csv.append(line)
	
	with open(cycles_post_location, 'w') as f:
		for line in compacted_csv:
			f.write(line)