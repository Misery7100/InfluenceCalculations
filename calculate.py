
import argparse
from classes.influence import Influence
import tensorflow as tf
from keras.applications import VGG16
import time

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Run Influences Calculation')

	parser.add_argument('--npz', '-z', default='final_cut',
                        type=str, required=True,
                        help='Compressed .npz file name with prepared data')

	parser.add_argument('--iter', '-it', default=30,
                        type=int, required=True,
                        help='Recursion depth for IHVP calculations')

	parser.add_argument('--bsize', '-bs', default=8,
                        type=int, required=True,
                        help='Batch size for IHVP calculations')

	parser.add_argument('--saveformat', '-sv', default='pdf',
                        type=str, required=False,
                        help='Output format for matplotlib figures')


	args = parser.parse_args() 					# parse args
	model = VGG16()								# create VGG16 instance
	influence = Influence(npz=args.npz+'.npz') 	# create Influence instance

	start_time = time.time()					# store start time

	# start calculations
	influence.calculate_all_influences(num_iter=args.iter, batch_size=args.bsize, save_format=args.saveformat)

	total_time = time.time() - start_time
        
	print(f'Calculation time: {(total_time // 60):.0f}min {(total_time % 60):.0f}s')