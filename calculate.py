
import argparse
from classes.influence import Influence
import tensorflow as tf
from keras.applications import VGG16
import time

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Run Influences Calculation')

	parser.add_argument('--iter', '-i', default=30,
                        type=int, required=True,
                        help='Recursion depth for IHVP calculations')

	parser.add_argument('--bsize', '-b', default=8,
                        type=int, required=True,
                        help='Batch size for IHVP calculations')

	parser.add_argument('--saveformat', '-sf', default='pdf',
                        type=str, required=False,
                        help='Output format for matplotlib figures')


	args = parser.parse_args()	# parse args
	model = VGG16()				# create VGG16 instance

	# create Influence instance

	influence = Influence(train_npz='train.npz', test_npz='test.npz', model=model) 

	start_time = time.time() # store start time

	# start calculations
	influence.calculate_all_influences(num_iter=args.iter, batch_size=args.bsize, save_format=args.saveformat)

	total_time = time.time() - start_time
      
    # print total spent time
	print(f'\nCalculation time: {(total_time // 60):.0f}min {(total_time % 60):.0f}s')