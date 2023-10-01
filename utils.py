import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from ML_utils import balance_data


def init_directories(w_path, plots_path, mod_path, np_arr_path):
    if not os.path.exists("./" + w_path):
        os.mkdir("./" + w_path)
    if not os.path.exists("./" + plots_path):
        os.mkdir("./" + plots_path)
    if not os.path.exists("./" + plots_path + "/anova_avg/"):
        os.mkdir("./" + plots_path + "/anova_avg/")
    if not os.path.exists("./" + plots_path + "/anova_avg/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_avg/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_avg/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_avg/som_no-bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_min/"):
        os.mkdir("./" + plots_path + "/anova_min/")
    if not os.path.exists("./" + plots_path + "/anova_min/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_min/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/anova_min/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/anova_min/som_no-bal_comp")
    if not os.path.exists("./" + mod_path):
        os.mkdir("./" + mod_path)
    if not os.path.exists("./" + mod_path + "/anova_avg/"):
        os.mkdir("./" + mod_path + "/anova_avg/")
    if not os.path.exists("./" + mod_path + "/anova_min/"):
        os.mkdir("./" + mod_path + "/anova_min/")
    if not os.path.exists("./" + np_arr_path):
        os.mkdir("./" + np_arr_path)
    if not os.path.exists("./" + np_arr_path + "/anova_avg/"):
        os.mkdir("./" + np_arr_path + "/anova_avg/")
    if not os.path.exists("./" + np_arr_path + "/anova_min/"):
        os.mkdir("./" + np_arr_path + "/anova_min/")


# load a file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.to_numpy()


def load_group(filename, numFeat, pathPrefix=""):
    loaded = list()
    for name in filename:
        data = load_file(pathPrefix + name)
        loaded.append(data)

    loaded = np.array(loaded[0][:, :numFeat])

    return loaded


def load_dataset_group(group, pathPrefix, numFeat):
    filepath = pathPrefix + group + "/"

    filename = list()
    # The “X_train.txt” file that contains the engineered features intended for fitting a model.
    filename += ["X_" + group + ".txt"]

    X = load_group(filename, numFeat, filepath)
    # The “y_train.txt” that contains the class labels for each observation (1-6).
    y = load_file(filepath + "/y_" + group + ".txt")

    return X, y


def load_dataset(pathPrefix, numFeat):
    # load train dataset
    trainX, trainy = load_dataset_group("train", pathPrefix + "/", numFeat)

    print(trainX.shape, trainy.shape)

    # load test dataset
    testX, testy = load_dataset_group("test", pathPrefix + "/", numFeat)

    print(testX.shape, testy.shape)

    # zero-offset class values to perform one-hot encode (default values 1-6)
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y

    trainy = tf.keras.utils.to_categorical(trainy)

    testy = tf.keras.utils.to_categorical(testy)

    if sys.argv[1] == "bal":
        return balance_data(trainX, trainy, testX, testy)
    else:
        return trainX, trainy, testX, testy
