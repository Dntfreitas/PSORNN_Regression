from ann import Ann


class ANNRedWine(Ann):
    def __init__(self):
        nInputs = 11
        nOutputs = 1
        nHidden = 10
        dataPath = 'datasets\wine\red.csv'
        super().__init__(nInputs, nOutputs, nHidden, dataPath)
