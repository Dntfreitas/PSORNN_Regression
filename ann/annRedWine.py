from ann.ann import Ann


class ANNRedWine(Ann):
    def __init__(self):
        super().__init__(11, 1, 'datasets/wine/red.csv')
