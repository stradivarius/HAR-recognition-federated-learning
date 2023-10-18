import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from ML_utils import balance_data
import pickle
from sklearn.model_selection import train_test_split



def init_directories(w_path, plots_path, mod_path, np_arr_path, dataset_type):
    if not os.path.exists("./" + plots_path):
        os.mkdir("./" + plots_path)
    if not os.path.exists("./" + plots_path + "/"+ dataset_type):
        os.mkdir("./" + plots_path + "/"+ dataset_type)
    if not os.path.exists("./" + plots_path + "/"+ dataset_type + "/anova_avg/"):
        os.mkdir("./" + plots_path + "/" + dataset_type + "/anova_avg/")
    if not os.path.exists("./" + plots_path + "/"+ dataset_type + "/anova_avg/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/" + dataset_type +  "/anova_avg/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/" + dataset_type +  "/anova_avg/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/" + dataset_type +  "/anova_avg/som_no-bal_comp")
    if not os.path.exists("./" + plots_path + "/" + dataset_type +  "/anova_min/"):
        os.mkdir("./" + plots_path + "/" + dataset_type +  "/anova_min/")
    if not os.path.exists("./" + plots_path + "/" + dataset_type +  "/anova_min/som_bal_comp/"):
        os.mkdir("./" + plots_path + "/" + dataset_type +  "/anova_min/som_bal_comp")
    if not os.path.exists("./" + plots_path + "/" + dataset_type +  "/anova_min/som_no-bal_comp/"):
        os.mkdir("./" + plots_path + "/" + dataset_type +  "/anova_min/som_no-bal_comp")

    if not os.path.exists("./" + mod_path):
        os.mkdir("./" + mod_path)
    if not os.path.exists("./" + mod_path + "/" + dataset_type):
        os.mkdir("./" + mod_path + "/" + dataset_type)
    if not os.path.exists("./" + mod_path + "/" + dataset_type +  "/anova_avg/"):
        os.mkdir("./" + mod_path + "/" + dataset_type +  "/anova_avg/")
    if not os.path.exists("./" + mod_path + "/" + dataset_type +  "/anova_min/"):
        os.mkdir("./" + mod_path + "/" + dataset_type +  "/anova_min/")

    if not os.path.exists("./" + np_arr_path):
        os.mkdir("./" + np_arr_path)
    if not os.path.exists("./" + np_arr_path + "/" + dataset_type):
        os.mkdir("./" + np_arr_path + "/" + dataset_type)
    if not os.path.exists("./" + np_arr_path + "/" + dataset_type +  "/anova_avg/"):
        os.mkdir("./" + np_arr_path + "/" + dataset_type +  "/anova_avg/")
    if not os.path.exists("./" + np_arr_path + "/" + dataset_type +  "/anova_min/"):
        os.mkdir("./" + np_arr_path + "/" + dataset_type + "/anova_min/")

    if not os.path.exists("./UCI HAR Dataset split/"):
        os.mkdir("./UCI HAR Dataset split/")
    if not os.path.exists("./UCI HAR Dataset split/train/"):
        os.mkdir("./UCI HAR Dataset split/train/")
    if not os.path.exists("./UCI HAR Dataset split/test"):
        os.mkdir("./UCI HAR Dataset split/test/")


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

def load_subject_dataset(subjects_number, pathPrefix):
    sub_map_train = load_file("./UCI HAR Dataset/train/subject_train.txt")
    sub_map_test = load_file("./UCI HAR Dataset/test/subject_test.txt")

    sub_map = np.concatenate((sub_map_train, sub_map_test))

    train_subjects = np.unique(sub_map)

    # caricare il dataset UCI e dividerlo per soggetti
    # carico e unisco il dataset (considerando 265 feature)
    uci_x_train, uci_y_train = load_dataset_group("train", pathPrefix, 265)
    uci_x_test, uci_y_test = load_dataset_group("test", pathPrefix, 265)

    X = np.concatenate((uci_x_train, uci_x_test))
    y = np.concatenate((uci_y_train, uci_y_test))

    trainX_lst = list()
    trainy_lst = list()
    testX_lst = list()
    testy_lst = list()

    for subject_index in range(int(subjects_number)):

        print("X_shape:", X.shape)
        print("y_shape:", y.shape)
        # prendo il dataset del soggetto corrispondente all'index
        datasetX, datasety = dataset_for_subject(X, y, sub_map, train_subjects[subject_index])
    
        # split subject dataset in 70% train and 30% test
        s_trainX, s_testX, s_trainy, s_testy = train_test_split(datasetX, datasety, train_size=0.70, shuffle=False)

        trainX_lst.append(s_trainX)
        trainy_lst.append(s_trainy)
        testX_lst.append(s_testX)
        testy_lst.append(s_testy)

        print("sub_trainX:", s_trainX.shape)
        print("sub_trainy:", s_trainy.shape)
        print("sub_testX:", s_testX.shape)
        print("sub_testy:", s_testy.shape)

        groups = ["train", "test"]
        # salvo il dataset in file csv
        for group in groups:
            if not os.path.exists("./UCI HAR Dataset split/"+ group +"/subject-" + str(subject_index)):
                os.mkdir("./UCI HAR Dataset split/"+ group +"/subject-" + str(subject_index))

        save_dataset(s_trainX, s_trainy, "./UCI HAR Dataset split/" + "train/" + "subject-" + str(subject_index) + "/", subject_index)
        save_dataset(s_testX, s_testy, "./UCI HAR Dataset split/" + "test/" + "subject-" + str(subject_index) + "/", subject_index)

    trainX = np.concatenate(trainX_lst, axis=0)
    trainy = np.concatenate(trainy_lst, axis=0)
    testX = np.concatenate(testX_lst, axis=0)
    testy = np.concatenate(testy_lst, axis=0)
    
    # zero-offset class values to perform one-hot encode (default values 1-6)
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y

    trainy = tf.keras.utils.to_categorical(trainy)

    testy = tf.keras.utils.to_categorical(testy)
    
    return  trainX, trainy, testX, testy



def load_uci_dataset(pathPrefix, numFeat):
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


def dataset_for_subject(Xtrain, ytrain, subjects_map, sub_id):
    # ottengo gli index delle righe corrispondenti al sub_id
    rows_indexes = [i for i in range(len(subjects_map)) if subjects_map[i]==sub_id]
    return Xtrain[rows_indexes, :], ytrain[rows_indexes]


def save_model(som, mod_path, typ, anova_val, som_dim, dataset_type):
    with open('./' + mod_path + "/" + dataset_type +  '/anova_' + typ + '/' + anova_val + '/som' + som_dim + 'x' + som_dim + '.p', 'wb') as outfile:
        pickle.dump(som, outfile)

def save_dataset(X, y, pathPrefix, sub_index):
    
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    # salvo il dataset del soggetto in un csv
    df_X.to_csv(pathPrefix + "subject-" + str(sub_index) + "-X.csv" , index=False, header=False)
    df_y.to_csv(pathPrefix + "subject-" + str(sub_index) + "-y.csv", index=False, header=False)