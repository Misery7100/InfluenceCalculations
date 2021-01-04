import numpy as np

class LabelSubset:
    def __init__(self, label):
        self.label = label

    def extractLabels(self, imgArr, imgLabArr):
        res = imgArr[np.where(imgLabArr == self.label)[0]]
        return res

    def subsetByLabels(self, arc):
        arr_x = arc.files[4::2]
        arr_y = arc.files[5::2]

        subs_dir = []

        for x, y in zip(arr_x, arr_y):
            exec(f'{x} = self.extractLabels(arc["{x}"], arc["{y}"])')
            exec(f'{y} = self.extractLabels(arc["{y}"], arc["{y}"])')

            subs_dir.append(f'{x}={x}')
            subs_dir.append(f'{y}={y}')

        subs_dir = ','.join(subs_dir)

        print(f'Creating label subset for {self.label} class... ', end='')
        print('finished')
        exec(f'np.savez("{self.label}",{subs_dir})')