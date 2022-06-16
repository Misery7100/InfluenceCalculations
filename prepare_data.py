import argparse
from src.folder_parser import FolderParser

# ------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Processing')

    parser.add_argument('--train', '-tr', default='train',
                        type=str, required=True,
                        help='Folder with train images')

    parser.add_argument('--test', '-ts', default='test',
                        type=str, required=True, nargs='+',
                        help='Folder with test images')

    parser.add_argument('--batchSize', '-bs', default='100',
                        type=int, required=True,
                        help='Batch size for train images')

    args = parser.parse_args()
    fp = FolderParser(args=args)
    fp.createZip()
