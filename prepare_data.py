
import argparse

from classes.folder_parser import FolderParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Processing')

    parser.add_argument('--train', '-tr', default='train',
                        type=str, required=True,
                        help='Folder with DNN train images')

    parser.add_argument('--test', '-ts', default='test',
                        type=str, required=True,
                        help='Folder with DNN test images')

    parser.add_argument('--modifiedImages', '-mod', default='orig styled texture wb',
                        type=str, required=True, nargs='+',
                        help='Folders with modified images, separated by space')

    parser.add_argument('--outName', '-o', default='final_cut',
                        required=True, type=str,
                        help='Output archive name without extension')

    args = parser.parse_args()
    fp = FolderParser(args=args)
    arraysInZip = fp.createZip()

    print(arraysInZip)
