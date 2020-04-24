import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Ann:

    def __init__(self, n_inputs, n_outputs, data_path):
        self.nInputs = n_inputs
        self.nOutputs = n_outputs
        self.dataPath = data_path

        # Data import
        self.dataSet = pd.read_csv(self.dataPath, sep=';')
        self.dataAttributes = self.dataSet.columns
        self.input = self.dataSet[self.dataAttributes[:self.nInputs]]
        self.output = self.dataSet[self.dataAttributes[self.nInputs + self.nOutputs - 1:]]

        # Split the data into training da test data set
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.input, self.output, test_size=0.3)

    def get_dimension(self):
        return self.dimensions

    def get_n_inputs(self):
        return self.nInputs

    def set_n_hidden(self, n_hidden):
        self.nHidden = n_hidden
        self.index = self.nInputs * self.nHidden
        self.dimensions = (self.nInputs * self.nHidden) + (self.nHidden * self.nOutputs)

    def compute(self, x):
        j = []
        for network in x:
            weights1 = network[:self.index].reshape(
                (self.nInputs, self.nHidden))
            weights2 = network[self.index:].reshape(
                (self.nHidden, self.nOutputs))
            # Hidden layer
            input_hidden = np.tanh(np.dot(self.X_train, weights1))
            # Output layer
            output = np.dot(input_hidden, weights2)
            # Compute MSE
            mse = mean_squared_error(self.Y_train, output)
            j.append(mse)
        return np.array(j)

    def test(self, x):
        weights1 = x[:self.index].reshape((self.nInputs, self.nHidden))
        weights2 = x[self.index:].reshape((self.nHidden, self.nOutputs))
        # Hidden layer
        input_hidden = np.tanh(np.dot(self.X_test, weights1))
        # Output layer
        output = np.dot(input_hidden, weights2)
        # Compute MSE
        mse = mean_squared_error(self.Y_test, output)
        r2 = r2_score(self.Y_test, output)
        return mse, r2
