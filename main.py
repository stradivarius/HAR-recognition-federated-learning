import pandas as pd
import sys
import numpy as np
import tensorflow as tf
import os
import random
import flwr as fl
import ray
import json


from logging import INFO, DEBUG
from flwr.common.logger import log
from typing import Dict, Tuple, List
from sklearn.metrics import classification_report
from minisom import MiniSom
from flwr.common import NDArrays, Scalar, Metrics
from utils import init_directories, load_subjects_dataset, create_subjects_datasets
from anovaf import get_anovaf
from plots import plot_fed_nofed_comp, plot_som_comp_dim
from plots import plot_som
from ML_utils import calculate_subjects_accs_mean

# input parameter: 
#   1) gen / load: genera il dataset dei singoli soggetti o carica quello già creato
#   2) num: numero di soggetti di cui prendere il dataset default = 2
#   3) centr: eseguire train centralizzato oppure no
#   4) fed: eseguire train federated oppure no
#   5) y / n: salva i grafici e i vari dati generati oppure non salvarli
#   6) num: dimensione minima della som
#   7) num: dimensione massima della som

# anova strutture di supporto
plot_labels_lst = []
accs_subjects_nofed = {}
accs_subjects_fed = {}
y = list()
new_y_test = list()


# default setup delle variabili di path e parametri
save_data = "y"
w_path = "weights UCI"
plots_path = "plots UCI"
mod_path = "som_models UCI"
np_arr_path = "np_arr UCI"
mean_path = "subjects_accs mean"
dataset_type = "split"
min_som_dim = 10
max_som_dim = 50
current_som_dim = min_som_dim
old_som_dim = 0
step = 10
exec_n = 1
total_execs = 0
actual_exec = 0
subjects_number = 2
centralized = False
federated = False
anova_type = "avg"
centr_type = sys.argv[3]
fed_type = ""

# check inputs parameter
fed_Xtrain = []
fed_ytrain = []
fed_Xtest = []
fed_ytest = []

subjects_number = sys.argv[2] 

if sys.argv[3] == "centr":
    centralized = True
    
    if sys.argv[4] == 'n':
        save_data = "n"

else:
    if sys.argv[4] == 'n':
        save_data = "n"

    fed_type = "no-fed"

if len(sys.argv) >= 8:
    min_som_dim = sys.argv[6]
    max_som_dim = sys.argv[7]


init_directories(w_path, plots_path, mod_path, np_arr_path, centr_type, fed_type, mean_path)


train_iter_lst = [100]  # , 250, 500, 750, 1000, 5000, 10000, 100000

divider = 10000  # cosa serve
range_lst = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # cosa serve
#range_lst = [8000]
    
total_execs = (
        (((max_som_dim + step) - min_som_dim) / step) * exec_n #* len(range_lst)
    )


#####FEDERATED FUNCTIONS

# Return the current local model parameters
def get_parameters(som):
    return [som.get_weights()]

def set_parameters(som, parameters):
    som._weights = parameters[0]

def classify_fed(som, data, X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap contiene una classificazione di campione in X_train 
    # con una delle classi in y (associazione neurone-label)
    
    new_y_train = list()
    for idx, item in enumerate(y_train):
        # inserisco in y gli index di ogni classe invertendo il one-hot encode
        new_y_train.append(np.argmax(y_train[idx]))

    winmap = som.labels_map(X_train , new_y_train)
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
    def __init__(self, som, Xtrain, ytrain, Xtest, ytest , train_iter, cid):
        self.som = som
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.train_iter = train_iter
        self.Xtest = Xtest
        self.ytest = ytest
        self.cid = cid

    # Return the current local model parameters
    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.som)
    
    # Receive model parameters from the server, 
    # train the model parameters on the local data, 
    # and return the (updated) model parameters to the server
    def fit(self, parameters, config):
        set_parameters(self.som, parameters)
        self.som.train_random(self.Xtrain, self.train_iter, verbose=False)
        return get_parameters(self.som), len(self.Xtrain), {}
    
    # Receive model parameters from the server,
    # evaluate the model parameters on the local data, 
    # and return the evaluation result to the server
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        new_y_test = list()
        for idx, item in enumerate(self.ytest):
            # inserisco in new_test_y gli index di ogni classe invertendo il one-hot encode
            new_y_test.append(np.argmax(self.ytest[idx]))

        set_parameters(self.som, parameters)
        class_report = classification_report(
            new_y_test,
            classify_fed(
                self.som,
                self.Xtest,
                self.Xtrain,
                self.ytrain
            ),
            zero_division=0.0,
            output_dict=True,
        )

        return float(0), len(self.Xtest), {"accuracy": float(class_report["accuracy"])}


def client_fn(cid) -> SomClient:
    neurons = current_som_dim
    train_iter = train_iter_lst[0]
    # prendo il dataset corrispondente al cid(client id)
    Xtrain = fed_Xtrain[int(cid)]
    ytrain = fed_ytrain[int(cid)]
    Xtest = fed_Xtest[int(cid)]
    ytest = fed_ytest[int(cid)]

    som = MiniSom(
            neurons,
            neurons,
            Xtrain.shape[1],
            sigma=5,
            learning_rate=0.1,
            neighborhood_function="gaussian",
            activation_distance="manhattan",
        )

    return SomClient(som, Xtrain, ytrain, Xtest, ytest, train_iter, cid)


def weighted_simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    log(DEBUG, f"current metrics: {metrics}")
    w_accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"w_accuracy": sum(w_accuracies) / sum(examples), "s_accuracy": sum(s_accuracies)/clients_num}

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(s_accuracies)/clients_num}
######


def train_federated(num_rounds):
    #definiamo come strategia FedAvg che 
    # FedAvg takes the 100 model updates and, as the name suggests, 
    # averages them. To be more precise, it takes the weighted average 
    # of the model updates, weighted by the number of examples each client used for training.
    #  The weighting is important to make sure that each data example has
    #  the same “influence” on the resulting global model. If one client 
    # has 10 examples, and another client has 100 examples,
    #  then - without weighting - each of the 10 examples would
    #  influence the global model ten times as much as each of the 100 examples.


   strategy = fl.server.strategy.FedAvg(
       fraction_fit=1.0,
       fraction_evaluate=1.0,
       min_fit_clients=int(subjects_number),
       min_evaluate_clients=int(subjects_number),
       min_available_clients=int(subjects_number),
       evaluate_metrics_aggregation_fn=simple_average,
   )
   client_resources = None
   hist = fl.simulation.start_simulation(
       client_fn = client_fn,
       num_clients = int(subjects_number),
       config = fl.server.ServerConfig(num_rounds=num_rounds),
       strategy = strategy,
       client_resources = client_resources,
   )
   #    ray_init_args = {"num_cpus": 2, "num_gpus":0.0}
   return hist.metrics_distributed


def train_nofederated(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld):
    global actual_exec
    global accs_subjects_nofed

    # deve essere eseguito il train e il test su
    # "subjects_number" dataset separati e poi bisogna salvarne la media
    for subj_idx, subj in enumerate(subjects_to_ld):
        print("trainX sub", trainX_lst[subj_idx].shape)
        print("trainy sub", trainy_lst[subj_idx].shape)
        print("testX sub", testX_lst[subj_idx].shape)
        print("testy sub", testy_lst[subj_idx].shape)
        accs_subjects_nofed.setdefault(subj,{})
        accs_subjects_nofed[subj].setdefault(current_som_dim, 0)
        actual_exec = 0
        run_training_nofed(trainX_lst[subj_idx], trainy_lst[subj_idx], testX_lst[subj_idx], testy_lst[subj_idx], subj)

def classify(som, data, X_train, y_train, neurons, train_iter, subj):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap contiene una classificazione di campione in X_train 
    # con una delle classi in y (associazione neurone-label)
    winmap = som.labels_map(X_train , y)
    default_class = np.sum( list (winmap.values())).most_common()[0][0]

    if save_data == 'y':
        final_map = {}

        for idx, val in enumerate(winmap):
            final_map.update({(val[0] * neurons) + val[1]: winmap[val].most_common()[0][0]})

        final_map_lst = []
        pos_count = 0
        w_tot = pow(neurons, 2)
        for i in range(w_tot):
            if i not in final_map:
                final_map.update({i: default_class})

        # inserisce l'associazione neurone label all'interno di
        # final_map_lst in ordine, in modo da far coincidere l'index di ogni classe
        # con il neurone(codificato attraverso la formula (val[0] * neurons) + val[1])
        while len(final_map_lst) < len(final_map):
            for idx, val in enumerate(final_map):
                if int(val) == pos_count:
                    final_map_lst.append(final_map[val])
                    pos_count += 1

        final_map_lst = np.array(final_map_lst)
        if not centralized:
                if not os.path.exists(
                "./" + np_arr_path +"/" + centr_type + "/" + fed_type + "/"+ "subject-" + str(subj) + "/"
            ):
                    os.mkdir(
                        "./"
                        + np_arr_path
                        +"/" + centr_type + "/" + fed_type
                        + "/subject-" + str(subj)
                        + "/"
                    )   
        if not os.path.exists('./' + np_arr_path + "/" + centr_type + "/" + fed_type  + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") ):
                os.mkdir('./' + np_arr_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") )
        
        np.savetxt('./' + np_arr_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "")  + '/map_lst_iter-' + str(train_iter) + '_' + "subjects-" + str(subjects_number) + "_" + 
                "avg" + '_' + str(neurons) + '.txt', final_map_lst, delimiter=' ')

    result = []
    for d in data :
        win_position = som.winner( d )
        if win_position in winmap :
            result.append( winmap [ win_position ].most_common()[0][0])
        else :
            result.append( default_class )
    return result


def execute_minisom_anova(
    X_train,
    y_train,
    X_test,
    y_test,
    neurons,
    train_iter,
    accs_tot_avg,
    subj,
):
    global old_som_dim
    global current_som_dim
    global exec_n
    global total_execs
    global actual_exec

    # calcolo risultati utilizzando diversi valori anova avg
    accuracies = []
    n_neurons = 0
    
    n_neurons = m_neurons = neurons
    som = MiniSom(
        n_neurons,
        m_neurons,
        X_train.shape[1],
        sigma=5,
        learning_rate=0.1,
        neighborhood_function="gaussian",
        activation_distance="manhattan",
    )
    som.random_weights_init(X_train)
    som.train_random(X_train, train_iter, verbose=False)  # random training
    #som.train_batch(X_lower_anova, train_iter, verbose=False)
    if save_data == 'y':
        if not centralized:
            if not os.path.exists('./' + mod_path + "/" + centr_type + "/" + fed_type + '/' + "subject-" + str(subj) + '/'):
                os.mkdir('./' + mod_path + "/" + centr_type + "/" + fed_type + '/' + "subject-" + str(subj) + '/')
        if not os.path.exists('./' + mod_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") ):
            os.mkdir('./' + mod_path + "/" + centr_type + "/" + fed_type + '/' + ( "subject-" + str(subj) + "/" if not centralized else "") )
    if not centralized:
        if not os.path.exists(
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + "/subject-" + str(subj) + "/"
        ):
            os.mkdir(
                "./"
                + plots_path
                +"/" + centr_type + "/" + fed_type
                + "/subject-" + str(subj) + "/"
            )
    
    if not os.path.exists(
        "./"
        + plots_path
        +"/" + centr_type + "/" + fed_type
        + ( "/subject-" + str(subj) + "/" if not centralized else "")
        + "/som_"
        + str(n_neurons)
    ):
        os.mkdir(
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + ( "/subject-" + str(subj) + "/" if not centralized else "")
            + "/som_"
            + str(n_neurons)
        )
    if save_data == "y":
        plot_som(
            som,
            X_train,
            y_train,
            "./"
            + plots_path
            +"/" + centr_type + "/" + fed_type
            + ( "/subject-" + str(subj) + "/" if not centralized else "")
            + "/som_"
            + str(n_neurons)
            + "/som_iter-"
            + str(train_iter)
            + "_plot_",
            X_train.shape[1],
            save_data,
            subjects_number,
            str(subj),
            centralized,
        )
    w = som.get_weights()
    #La notazione -1 in una delle dimensioni indica a NumPy di inferire
    #automaticamente la dimensione in modo tale da mantenere il numero 
    #totale di elementi invariato. In questo caso, viene inferito in modo 
    #tale da mantenere il numero di elementi nella terza dimensione 
    #(l'ultimo elemento di w.shape) invariato.
    w = w.reshape((-1, w.shape[2]))
    #if not old_som_dim == current_som_dim:
    if save_data == "y":
        if not centralized:
            if not os.path.exists(
            "./" + np_arr_path +"/" + centr_type + "/" + fed_type + "/"+ "subject-" + str(subj) + "/"
        ):
                os.mkdir(
                    "./"
                    + np_arr_path
                    +"/" + centr_type + "/" + fed_type
                    + "/subject-" + str(subj)
                    + "/"
                )   
            
        if not os.path.exists(
            "./" + np_arr_path +"/" + centr_type + "/" + fed_type + "/"+  ( "subject-" + str(subj) + "/" if not centralized else "") 
        ):
            os.mkdir(
                "./"
                + np_arr_path
                +"/" + centr_type + "/" + fed_type
                + "/"
                + ( "subject-" + str(subj) + "/" if not centralized else "")
            )   
        np.savetxt(
            "./"
            + np_arr_path
            +"/" + centr_type + "/" + fed_type
            + "/"
            + ( "subject-" + str(subj) + "/" if not centralized else "")
            + "weights_lst_avg_iter-"
            + str(train_iter)
            + "_"
            + "subjects-" + str(subjects_number)
            + "_"
            + str(neurons)
            + ".txt",
            w,
            delimiter=" ",
        )
        if not centralized:
            if not os.path.exists(
            "./" + mod_path +"/" + centr_type + "/" + fed_type + "/" + "subject-" + str(subj) + "/"
        ):
                os.mkdir(
                    "./" + mod_path +"/" + centr_type + "/" + fed_type + "/" + "subject-" + str(subj) + "/"
                )
        if not os.path.exists(
            "./" + mod_path +"/" + centr_type + "/" + fed_type + "/" + ( "subject-" + str(subj) + "/" if not centralized else "") 
        ):
            os.mkdir(
                "./" + mod_path +"/" + centr_type + "/" + fed_type + "/" + ( "subject-" + str(subj) + "/" if not centralized else "")
            )
        #old_som_dim = current_som_dim
    # esegue una divisione per zero quando
    # un label non è presente tra quelli predetti
    class_report = classification_report(
        new_y_test,
        classify(
            som,
            X_test,
            X_train,
            y_train,
            n_neurons,
            train_iter,
            subj,
        ),
        zero_division=0.0,
        output_dict=True,
    )
    #save_model(som, mod_path, "avg", str(a_val / divider), str(n_neurons), centr_type, fed_type)
    
    accuracies.append(class_report["accuracy"])
    # insert in accuracy dictionary the accuracy for anova val
    accs_tot_avg.append(class_report["accuracy"])
    if not centralized:
        accs_subjects_nofed[subj][n_neurons] = class_report["accuracy"]

    actual_exec += 1
    percentage = round((actual_exec / total_execs) * 100, 2)
    print("\rProgress: ", percentage, "%", end="")



def run_training(trainX, trainy, testX, testy, subj=0):

    # som preparation
    
    global current_som_dim
    global range_lst


    print("trainX", trainX.shape)
    print("trainy", trainy.shape)
    print("testX", testX.shape)
    print("testy", testy.shape)


    y.clear()
    new_y_test.clear()
    for idx, item in enumerate(trainy):
        # inserisco in y gli index di ogni classe invertendo il one-hot encode
        y.append(np.argmax(trainy[idx]))

    for idx, item in enumerate(testy):
        # inserisco in new_test_y gli index di ogni classe invertendo il one-hot encode
        new_y_test.append(np.argmax(testy[idx]))
    
    for t_iter in train_iter_lst:
        plot_labels_lst.clear()

        # dizionario accuracies per varie dimensioni della som e valori anova
        accs_mean = {10: 0.0}
        accs_max = {10: 0.0}
        accs_min = {10: 0.0}

        # for da 10 a 60 per le varie dimensioni delle som
        for i in range(min_som_dim, max_som_dim + step, step):
            # setup valori anova del dizionario delle accuracies per il dataset UCI
            if not centralized:
                accs_subjects_nofed[subj].update({i: 0})

        for i in range(min_som_dim, max_som_dim + step, step):
            current_som_dim = i
            accs_tot = []

            plot_labels_lst.append(str(i) + "x" + str(i))
            for j in range(1, exec_n + 1, 1):
                execute_minisom_anova(
                    X_train=trainX,
                    y_train=trainy,
                    X_test=testX,
                    y_test=testy,
                    neurons=i,
                    train_iter=t_iter,
                    accs_tot_avg=accs_tot,
                    subj=subj,
                )
            accs_tot_min = np.min(accs_tot)
            accs_tot_max = np.max(accs_tot)
            accs_tot_mean = np.mean(accs_tot)

            accs_mean.update({i: accs_tot_mean})
            accs_max.update({i: accs_tot_max})
            accs_min.update({i: accs_tot_min})

        plot_som_comp_dim(t_iter, accs_mean, accs_max, accs_min, plot_labels_lst, save_data, centr_type, fed_type, subjects_number, plots_path, exec_n, str(subj), centralized)
        
       # plot_som_comp(
       #     t_iter,
       #     accs_mean,
       #     accs_max,
       #     accs_min,
       #     plot_labels_lst,
       #     save_data,
       #     centr_type,
       #     fed_type,
       #     subjects_number,
       #     plots_path,
       #     range_lst,
       #     divider,
       #     exec_n,
       #     str(subj),
       #     centralized
       # )

def run_training_nofed(trainX, trainy, testX, testy, subj=0):
    y.clear()
    new_y_test.clear()
    global accs_subjects_nofed

    for idx, item in enumerate(trainy):
        # inserisco in y gli index di ogni classe invertendo il one-hot encode
        y.append(np.argmax(trainy[idx]))

    for idx, item in enumerate(testy):
        # inserisco in new_test_y gli index di ogni classe invertendo il one-hot encode
        new_y_test.append(np.argmax(testy[idx]))
    
    accs_tot = []
    for j in range(1, exec_n + 1, 1):
                execute_minisom_anova(
                    X_train=trainX,
                    y_train=trainy,
                    X_test=testX,
                    y_test=testy,
                    neurons=current_som_dim,
                    train_iter=train_iter_lst[0],
                    accs_tot_avg=accs_tot,
                    subj=subj,
                )


def run():
    global actual_exec
    global current_som_dim
    global fed_Xtrain
    global fed_ytrain
    global fed_Xtest
    global fed_ytest
    global accs_subjects_nofed
    global accs_subjects_fed

    # Use np.concatenate() Function
    
    if sys.argv[1] == "gen":
        create_subjects_datasets(False)
    elif sys.argv[1] == "a-gen":
        create_subjects_datasets(True)

    subjects_to_ld=random.sample(range(1, 31), int(subjects_number))

    accs_subjects_nofed.update()

    if centralized:
        # se si sceglie l'esecuzione centralizzata
        # il train viene eseguito su un dataset composto 
        # dall'insieme dei dataset di "subjects_number" soggetti
        trainX, trainy, testX, testy = load_subjects_dataset(subjects_to_ld, "concat")

        run_training(trainX, trainy, testX, testy)
    else:
    
        trainX_lst, trainy_lst, testX_lst, testy_lst = load_subjects_dataset(subjects_to_ld, "separated")

        fed_Xtrain = trainX_lst
        fed_ytrain = trainy_lst
        fed_Xtest = testX_lst
        fed_ytest = testy_lst

        for dim in range(min_som_dim, max_som_dim + step, step):
            current_som_dim = dim

            accs_fed = train_federated(20) 
            accs_subjects_fed.update({current_som_dim: accs_fed})
        
            train_nofederated(trainX_lst, trainy_lst, testX_lst, testy_lst, subjects_to_ld)
            
        calculate_subjects_accs_mean(accs_subjects_nofed, accs_subjects_fed, min_som_dim, max_som_dim, step, mean_path, centr_type, subjects_to_ld)
        plot_fed_nofed_comp(mean_path, centr_type, min_som_dim, max_som_dim, step)



if __name__ == "__main__":
    run()
