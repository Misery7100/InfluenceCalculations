import argparse
from src.folder_parser import FolderParser

# ------------------------- #

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Image Processing')

    parser.add_argument('--test', '-ts', default='test',
                        type=str, required=True, nargs='+',
                        help='Folder with test images')

    args = parser.parse_args()
    fp = FolderParser(args=args)
    fp.createTest((224, 224))
