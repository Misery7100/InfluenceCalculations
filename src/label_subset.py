import numpy as np

# ------------------------- #

class LabelSubset:

    def __init__(self, label, folder=''):
        self.label = label
        self.folder = folder

    # ------------------------- #

    def extractLabels(self, imgArr, imgLabArr):
        res = imgArr[np.where(imgLabArr == self.label)[0]]
        return res
    
    # ------------------------- #

    def subsetByLabels(self, x, y):

        x = self.extractLabels(x, y)
        y = self.extractLabels(y, y)

        print(f'Create label subset for {self.label} class... ', end='')

        np.savez(f"{self.folder}/{self.label}", x=x, y=y)

        print('finished')