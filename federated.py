from typing import Dict, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar, FlowerClient
import numpy as np
from sklearn.metrics import classification_report
from minisom import MiniSom


def get_parameters(som):
    return som.get_weights()

def set_parameters(som, parameters):
    som._weights = parameters

def classify_fed(som, data, X_train, y):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap contiene una classificazione di campione in X_train 
    # con una delle classi in y (associazione neurone-label)
    winmap = som.labels_map(X_train , y)
    default_class = np.sum( list (winmap.values())).most_common()[0][0]
    
    result = []
    for d in data :
        win_position = som.winner( d )
        if win_position in winmap :
            result.append( winmap [ win_position ].most_common()[0][0])
        else :
            result.append( default_class )
    return result

class SomClient(fl.client.NumPyClient):
    def __init__(self, som, Xtrain, Xtest, train_iter, new_y_train, new_y_test):
        self.som = som
        self.Xtrain = Xtrain
        self.train_iter = train_iter
        self.Xtest = Xtest
        self.new_y_train = new_y_train
        self.new_y_test = new_y_test

    
    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.som)
    
    def fit(self, parameters, config):
        set_parameters(self.som, parameters)
        self.som.train_random(self.Xtrain, self.train_iter, verbose=False)
        return get_parameters(self.som), len(self.Xtrain), {}
    
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.som, parameters)
        class_report = classification_report(
            self.new_y_test,
            classify_fed(
                self.som,
                self.Xtest,
                self.Xtrain,
            ),
            zero_division=0.0,
            output_dict=True,
        )

        return float(0), len(self.Xtest), {"accuracy": float(class_report["accuracy"])}


def client_fn(cid) -> FlowerClient:

    

    som = MiniSom(
            neurons,
            neurons,
            Xtrain.shape[1],
            sigma=5,
            learning_rate=0.1,
            neighborhood_function="gaussian",
            activation_distance="manhattan",
        )
    
