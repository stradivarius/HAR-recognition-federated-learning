import pandas as pd
import sys
import numpy as np
import tensorflow as tf
from minisom import MiniSom
import os
from sklearn.metrics import classification_report


from utils import init_directories, load_uci_dataset, save_model, load_subject_dataset, load_dataset_group, load_file
from anovaf import get_anovaf
from plots import plot_som_comp
from plots import plot_som
from ML_utils import balance_data

# anova strutture di supporto
acc_anova_avg_lst = []
acc_anova_min_lst = []
n_feat_anova_avg_lst = []
n_feat_anova_min_lst = []
anova_val_tested_global = []
plot_labels_lst = []
anova_nof_avg_global = []
anova_acc_avg_global = []
y = list()
new_y_test = list()


# default setup delle variabili di path e parametri
save_data = "y"
w_path = "weights UCI"
plots_path = "plots UCI"
mod_path = "som_models UCI"
np_arr_path = "np_arr UCI"
min_som_dim = 10
max_som_dim = 30
current_som_dim = min_som_dim
old_som_dim = 0
step = 10
exec_n = 1
total_execs = 0
actual_exec = 0
subjects_number = 1


# check inputs parameter
    
if sys.argv[3] == 'n':
    save_data = "n"

if len(sys.argv) >= 6:
    subjects_number = sys.argv[5] 

if len(sys.argv) >= 8:
    min_som_dim = sys.argv[6]
    max_som_dim = sys.argv[7]

if len(sys.argv) >= 9:
    exec_n = sys.argv[8]


init_directories(w_path, plots_path, mod_path, np_arr_path)


train_iter_lst = [6000]  # , 250, 500, 750, 1000, 5000, 10000, 100000

divider = 10000  # cosa serve
range_lst = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]  # cosa serve

total_execs = (
    (((max_som_dim + step) - min_som_dim) / step) * exec_n * len(range_lst) * 2
)
if sys.argv[2] == "avg" or sys.argv[2] == "min":
    total_execs = (
        (((max_som_dim + step) - min_som_dim) / step) * exec_n * len(range_lst)
    )
        


def classify(som, data, X_train, y_train, neurons, typ, a_val, train_iter):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # perchè viene usato y e non y_train
    winmap = som.labels_map(X_train , y)
    default_class = np.sum( list (winmap.values())).most_common()[0][0]

    if save_data == 'y':
        if not os.path.exists('./' + mod_path + '/anova_' + typ + '/' + str(a_val) + '/'):
            os.mkdir('./' + mod_path + '/anova_' + typ + '/' + str(a_val) + '/')
        final_map = {}

        for idx, val in enumerate(winmap):
            final_map.update({(val[0] * neurons) + val[1]: winmap[val].most_common()[0][0]})
    
        final_map_lst = []
        pos_count = 0
        w_tot = pow(neurons, 2)
        for i in range(w_tot):
            if i not in final_map:
                final_map.update({i: default_class})
        
        while len(final_map_lst) < len(final_map):
            for idx, val in enumerate(final_map):
                if int(val) == pos_count:
                    final_map_lst.append(final_map[val])
                    pos_count += 1
        final_map_lst = np.array(final_map_lst)
        if not os.path.exists('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/'):
                os.mkdir('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/')
        np.savetxt('./' + np_arr_path + '/anova_' + typ + '/' + str(a_val) + '/map_lst_iter-' + str(train_iter) + '_' +
                       sys.argv[2] + '_' + str(neurons) + '.txt', final_map_lst, delimiter=' ')

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
    count_anim,
    accs_tot_avg,
    accs_tot_min,
    varianza_media_classi,
    varianza_min_classi,
):
    global old_som_dim
    global current_som_dim
    global exec_n
    global total_execs
    global actual_exec

    if sys.argv[2] == "avg" or sys.argv[2] == "avgmin":
        # calcolo risultati utilizzando diversi valori anova avg
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        n_neurons = 0
        # in base a cosa sono stati calcolati i range per selezionare i valori anova?
        for a_val in range_lst:
            less_than_anova_vals = []
            greater_than_anova_vals = []
            # si sceglie l'index delle feature che andranno a comporre l'input del modello
            for idx, val in enumerate(varianza_media_classi):
                if val > a_val / divider:
                    greater_than_anova_vals.append(idx)
                else:
                    less_than_anova_vals.append(idx)

            # chiedere se per ogni osservazione si selezionano le feature minori ai valori anova tramite index
            X_lower_anova = X_train[:, less_than_anova_vals]
            X_greater_anova = X_train[:, greater_than_anova_vals]

            n_neurons = m_neurons = neurons

            som = MiniSom(
                n_neurons,
                m_neurons,
                X_lower_anova.shape[1],
                sigma=5,
                learning_rate=0.1,
                neighborhood_function="gaussian",
                activation_distance="manhattan",
            )

           
            som.random_weights_init(X_lower_anova)
            som.train_random(X_lower_anova, train_iter, verbose=False)  # random training

            if save_data == 'y':
                if not os.path.exists('./' + mod_path + '/anova_' + sys.argv[2] + '/' + str(a_val / divider) + '/'):
                    os.mkdir('./' + mod_path + '/anova_' + sys.argv[2] + '/' + str(a_val / divider) + '/')

            if not os.path.exists(
                "./"
                + plots_path
                + "/anova_avg/som_"
                + sys.argv[1]
                + "_"
                + str(n_neurons)
            ):
                os.mkdir(
                    "./"
                    + plots_path
                    + "/anova_avg/som_"
                    + sys.argv[1]
                    + "_"
                    + str(n_neurons)
                )
            if save_data == "y":
                plot_som(
                    som,
                    X_lower_anova,
                    y_train,
                    "./"
                    + plots_path
                    + "/anova_avg/som_"
                    + sys.argv[1]
                    + "_"
                    + str(n_neurons)
                    + "/som_iter-"
                    + str(train_iter)
                    + "_plot_",
                    a_val / divider,
                    X_lower_anova.shape[1],
                    count_anim,
                    save_data,
                )
            w = som.get_weights()
        
            #La notazione -1 in una delle dimensioni indica a NumPy di inferire
            #automaticamente la dimensione in modo tale da mantenere il numero 
            #totale di elementi invariato. In questo caso, viene inferito in modo 
            #tale da mantenere il numero di elementi nella terza dimensione 
            #(l'ultimo elemento di w.shape) invariato.
            w = w.reshape((-1, w.shape[2]))

            #if not old_som_dim == current_som_dim:
            print("old:", old_som_dim)
            print("current:", current_som_dim)
            if save_data == "y":
                if not os.path.exists(
                    "./" + np_arr_path + "/anova_avg/" + str(a_val / divider) + "/"
                ):
                    os.mkdir(
                        "./"
                        + np_arr_path
                        + "/anova_avg/"
                        + str(a_val / divider)
                        + "/"
                    )
                np.savetxt(
                    "./"
                    + np_arr_path
                    + "/anova_avg/"
                    + str(a_val / divider)
                    + "/weights_lst_avg_iter-"
                    + str(train_iter)
                    + "_"
                    + sys.argv[1]
                    + "_"
                    + str(neurons)
                    + ".txt",
                    w,
                    delimiter=" ",
                )
                if not os.path.exists(
                    "./" + mod_path + "/anova_avg/" + str(a_val / divider) + "/"
                ):
                    os.mkdir(
                        "./" + mod_path + "/anova_avg/" + str(a_val / divider) + "/"
                    )
                #old_som_dim = current_som_dim

            class_report = classification_report(
                new_y_test,
                classify(
                    som,
                    X_test[:, less_than_anova_vals],
                    X_lower_anova,
                    y_train,
                    n_neurons,
                    "avg",
                    a_val / divider,
                    train_iter,
                ),
                output_dict=True,
            )

            save_model(som, mod_path, sys.argv[2], str(a_val / divider), str(n_neurons))
           
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_lower_anova.shape[1])
            accuracies.append(class_report["accuracy"])
            # insert in accuracy dictionary the accuracy for anova val
            accs_tot_avg[a_val / divider].append(class_report["accuracy"])
            actual_exec += 1
            percentage = round((actual_exec / total_execs) * 100, 2)
            print("\rProgress: ", percentage, "%", end="")

                
            acc_anova_avg_lst.append(accuracies)
            n_feat_anova_avg_lst.append(n_feature_per_aval)

            #plt.figure()
            #plt.plot(anova_val_tested_str, accuracies, marker="o")
            #plt.xlabel("Anova Threshold")
            #plt.ylabel("mean of accuracies on 10 executions")
            #plt.title(
            #    "Accuracies comparison choosing the mean of the variances per class per f."
            #)
            #plt.close()
            #plt.bar(anova_val_tested_str, n_feature_per_aval)
            #plt.xlabel("anova val")
            #plt.ylabel("n° features")
            #plt.title(
            #    "N° of features comparison choosing the mean of the variances per class per f."
            #)
            #plt.close()

    if sys.argv[2] == "min" or sys.argv[2] == "avgmin":
        # calcolo risultati utilizzando diversi valori anova avg
        anova_val_tested = []
        anova_val_tested_str = []
        n_feature_per_aval = []
        accuracies = []
        n_neurons = 0
        # in base a cosa sono stati calcolati i range per selezionare i valori anova?
        for a_val in range_lst:
            less_than_anova_vals = []
            greater_than_anova_vals = []
            # si sceglie l'index delle feature che andranno a comporre l'input del modello
            for idx, val in enumerate(varianza_min_classi):
                if val > a_val / divider:
                    greater_than_anova_vals.append(idx)
                else:
                    less_than_anova_vals.append(idx)

            # chiedere se per ogni osservazione si selezionano le feature minori ai valori anova tramite index
            X_lower_anova = X_train[:, less_than_anova_vals]
            X_greater_anova = X_train[:, greater_than_anova_vals]

            n_neurons = m_neurons = neurons

            som = MiniSom(
                n_neurons,
                m_neurons,
                X_lower_anova.shape[1],
                sigma=5,
                learning_rate=0.1,
                neighborhood_function="gaussian",
                activation_distance="manhattan",
            )

           
            som.random_weights_init(X_lower_anova)
            som.train_random(X_lower_anova, train_iter, verbose=False)  # random training

            if save_data == 'y':
                if not os.path.exists('./' + mod_path + '/anova_' + sys.argv[2] + '/' + str(a_val / divider) + '/'):
                    os.mkdir('./' + mod_path + '/anova_' + sys.argv[2] + '/' + str(a_val / divider) + '/')

            if not os.path.exists(
                "./"
                + plots_path
                + "/anova_min/som_"
                + sys.argv[1]
                + "_"
                + str(n_neurons)
            ):
                os.mkdir(
                    "./"
                    + plots_path
                    + "/anova_min/som_"
                    + sys.argv[1]
                    + "_"
                    + str(n_neurons)
                )
            if save_data == "y":
                plot_som(
                    som,
                    X_lower_anova,
                    y_train,
                    "./"
                    + plots_path
                    + "/anova_min/som_"
                    + sys.argv[1]
                    + "_"
                    + str(n_neurons)
                    + "/som_iter-"
                    + str(train_iter)
                    + "_plot_",
                    a_val / divider,
                    X_lower_anova.shape[1],
                    count_anim,
                    save_data,
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
                if not os.path.exists(
                    "./" + np_arr_path + "/anova_min/" + str(a_val / divider) + "/"
                ):
                    os.mkdir(
                        "./"
                        + np_arr_path
                        + "/anova_min/"
                        + str(a_val / divider)
                        + "/"
                    )
                np.savetxt(
                    "./"
                    + np_arr_path
                    + "/anova_min/"
                    + str(a_val / divider)
                    + "/weights_lst_min_iter-"
                    + str(train_iter)
                    + "_"
                    + sys.argv[1]
                    + "_"
                    + str(neurons)
                    + ".txt",
                    w,
                    delimiter=" ",
                )
                if not os.path.exists(
                    "./" + mod_path + "/anova_min/" + str(a_val / divider) + "/"
                ):
                    os.mkdir(
                        "./" + mod_path + "/anova_min/" + str(a_val / divider) + "/"
                    )
                #old_som_dim = current_som_dim

            class_report = classification_report(
                new_y_test,
                classify(
                    som,
                    X_test[:, less_than_anova_vals],
                    X_lower_anova,
                    y_train,
                    n_neurons,
                    "min",
                    a_val / divider,
                    train_iter,
                ),
                output_dict=True,
            )

            save_model(som, mod_path, sys.argv[2], str(a_val / divider), str(n_neurons))
           
            anova_val_tested.append(a_val / divider)
            anova_val_tested_str.append(str(a_val / divider))
            n_feature_per_aval.append(X_lower_anova.shape[1])
            accuracies.append(class_report["accuracy"])
            # insert in accuracy dictionary the accuracy for anova val
            accs_tot_min[a_val / divider].append(class_report["accuracy"])
            actual_exec += 1
            percentage = round((actual_exec / total_execs) * 100, 2)
            print("\rProgress: ", percentage, "%", end="")

                
            acc_anova_min_lst.append(accuracies)
            n_feat_anova_min_lst.append(n_feature_per_aval)

            #plt.figure()
            #plt.plot(anova_val_tested_str, accuracies, marker="o")
            #plt.xlabel("Anova Threshold")
            #plt.ylabel("mean of accuracies on 10 executions")
            #plt.title(
            #    "Accuracies comparison choosing the mean of the variances per class per f."
            #)
            #plt.close()
            #plt.bar(anova_val_tested_str, n_feature_per_aval)
            #plt.xlabel("anova val")
            #plt.ylabel("n° features")
            #plt.title(
            #    "N° of features comparison choosing the mean of the variances per class per f."
            #)
            #plt.close()

def run_training(trainX, trainy, testX, testy):

    print("trainX:", trainX.shape)
    print("trainy:", trainy.shape)
    print("testX:", testX.shape)
    print("testy:", testy.shape)

    # som preparation
    ##############################
    count_anim = 0
    global current_som_dim
    for idx, item in enumerate(trainy):
        # inserisco in y gli index di ogni classe 
        y.append(np.argmax(trainy[idx]))

    for idx, item in enumerate(testy):
        # inserisco in new_test_y gli index di ogni classe
        new_y_test.append(np.argmax(testy[idx]))
    ############################## capire a che serve

    # perchè train iter è a 2?
    for t_iter in train_iter_lst:
        acc_anova_avg_lst.clear()
        acc_anova_min_lst.clear()
        n_feat_anova_avg_lst.clear()
        n_feat_anova_min_lst.clear()
        plot_labels_lst.clear()
        anova_nof_avg_global.clear()
        anova_acc_avg_global.clear()

        # calcolo varianza media e minima delle classi tramite ANOVA-F
        var_avg_c, var_min_c = get_anovaf(trainX, trainy, testX, testy)

        # dizionario accuracies per varie dimensioni della som e valori anova
        accs_min_mean = {10: {}}
        accs_min_max = {10: {}}
        accs_min_min = {10: {}}
        accs_avg_mean = {10: {}}
        accs_avg_max = {10: {}}
        accs_avg_min = {10: {}}

        if sys.argv[2] == "avg":
            # for da 10 a 60 per le varie dimensioni delle som
            for i in range(min_som_dim, max_som_dim + step, step):
                # setup valori anova del dizionario delle accuracies per il dataset UCI
                accs_min_mean.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_min_max.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_min_min.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_mean.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_max.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_min.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )

            for i in range(min_som_dim, max_som_dim + step, step):
                current_som_dim = i
                accs_tot_min = {
                    0.1: [],
                    0.2: [],
                    0.3: [],
                    0.4: [],
                    0.5: [],
                    0.6: [],
                    0.7: [],
                    0.8: [],
                    0.9: [],
                    1.0: [],
                }
                accs_tot_avg = {
                    0.1: [],
                    0.2: [],
                    0.3: [],
                    0.4: [],
                    0.5: [],
                    0.6: [],
                    0.7: [],
                    0.8: [],
                    0.9: [],
                    1.0: [],
                }
                plot_labels_lst.append(str(i) + "x" + str(i))
                for j in range(1, exec_n + 1, 1):
                    execute_minisom_anova(
                        X_train=trainX,
                        y_train=trainy,
                        X_test=testX,
                        y_test=testy,
                        neurons=i,
                        train_iter=t_iter,
                        count_anim=count_anim,
                        accs_tot_avg=accs_tot_avg,
                        accs_tot_min=accs_tot_min,
                        varianza_media_classi=var_avg_c,
                        varianza_min_classi=var_min_c,
                    )

                accs_tot_min_min = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }
                accs_tot_avg_min = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }
                accs_tot_min_max = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }
                accs_tot_avg_max = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }
                accs_tot_min_mean = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }
                accs_tot_avg_mean = {
                    0.1: 0.0,
                    0.2: 0.0,
                    0.3: 0.0,
                    0.4: 0.0,
                    0.5: 0.0,
                    0.6: 0.0,
                    0.7: 0.0,
                    0.8: 0.0,
                    0.9: 0.0,
                    1.0: 0.0,
                }

                for key in accs_tot_avg.keys():
                    accs_tot_avg_mean.update({key: np.mean(accs_tot_avg[key])})
                    accs_tot_avg_max.update({key: np.max(accs_tot_avg[key])})
                    accs_tot_avg_min.update({key: np.min(accs_tot_avg[key])})

                accs_avg_mean.update({i: accs_tot_avg_mean})
                accs_avg_max.update({i: accs_tot_avg_max})
                accs_avg_min.update({i: accs_tot_avg_min})
        else:
            for i in range(min_som_dim, max_som_dim + step, step):
                accs_min_mean.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_min_max.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_min_min.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_mean.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_max.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
                accs_avg_min.update(
                    {
                        i: {
                            0.1: [],
                            0.2: [],
                            0.3: [],
                            0.4: [],
                            0.5: [],
                            0.6: [],
                            0.7: [],
                            0.8: [],
                            0.9: [],
                            1.0: [],
                        }
                    }
                )
            for i in range(min_som_dim, max_som_dim + step, step):
                    current_som_dim = i
                    accs_tot_min = {
                        0.1: [],
                        0.2: [],
                        0.3: [],
                        0.4: [],
                        0.5: [],
                        0.6: [],
                        0.7: [],
                        0.8: [],
                        0.9: [],
                        1.0: [],
                    }
                    accs_tot_avg = {
                        0.1: [],
                        0.2: [],
                        0.3: [],
                        0.4: [],
                        0.5: [],
                        0.6: [],
                        0.7: [],
                        0.8: [],
                        0.9: [],
                        1.0: [],
                    }
                    plot_labels_lst.append(str(i) + "x" + str(i))

                    for j in range(1, exec_n + 1, 1):
                        execute_minisom_anova(
                            X_train=trainX,
                            y_train=trainy,
                            X_test=testX,
                            y_test=testy,
                            neurons=i,
                            train_iter=t_iter,
                            count_anim=count_anim,
                            accs_tot_avg=accs_tot_avg,
                            accs_tot_min=accs_tot_min,
                            varianza_media_classi=var_avg_c,
                            varianza_min_classi=var_min_c,
                        )

                    accs_tot_min_min = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }
                    accs_tot_avg_min = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }
                    accs_tot_min_max = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }
                    accs_tot_avg_max = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }
                    accs_tot_min_mean = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }
                    accs_tot_avg_mean = {
                        0.1: 0.0,
                        0.2: 0.0,
                        0.3: 0.0,
                        0.4: 0.0,
                        0.5: 0.0,
                        0.6: 0.0,
                        0.7: 0.0,
                        0.8: 0.0,
                        0.9: 0.0,
                        1.0: 0.0,
                    }

                    for k in accs_tot_avg.keys():
                        accs_tot_min_mean.update({k: np.mean(accs_tot_min[k])})
                        accs_tot_min_max.update({k: np.max(accs_tot_min[k])})
                        accs_tot_min_min.update({k: np.min(accs_tot_min[k])})
                    accs_min_mean.update({i: accs_tot_min_mean})
                    accs_min_max.update({i: accs_tot_min_max})
                    accs_min_min.update({i: accs_tot_min_min})
        plot_som_comp(
            t_iter,
            accs_avg_mean,
            accs_min_mean,
            accs_avg_max,
            accs_min_max,
            accs_avg_min,
            accs_min_min,
            plot_labels_lst,
            save_data,
            plots_path,
            range_lst,
            divider,
            exec_n,
        )
        count_anim += 1


def run():
    dataset = "UCI HAR Dataset"

    if sys.argv[4] == 'full':
        trainX, trainy, testX, testy = load_uci_dataset("./" + dataset, 265)

        print("trainx:",trainX.shape)
        print("trainy:",trainy.shape)
        print("testx", testX.shape)
        print("testy", testy.shape)

        run_training(trainX, trainy, testX, testy)
    
    elif sys.argv[4] == 'split':   
        uci_trainX, uci_trainy = load_dataset_group("train", "./" + dataset + "/", 265)
        sub_map = load_file("./UCI HAR Dataset/train/subject_train.txt")

        print("ucix:", uci_trainX.shape)
        print("uciy:", uci_trainy.shape)
        for i in range(int(subjects_number)):
         

            trainX, trainy, testX, testy = load_subject_dataset(i, uci_trainX, uci_trainy, sub_map)

            print("sub_trainx:",trainX.shape)
            print("sub_trainy:",trainy.shape)
            print("sub_testx", testX.shape)
            print("sub_testy", testy.shape)

            # balancing dataset
            if sys.argv[1] == "bal":
                trainX, trainy, testX, testy =  balance_data(trainX, trainy, testX, testy)
            
            print("bal sub_trainx:",trainX.shape)
            print("bal sub_trainy:",trainy.shape)
            print("bal sub_testx", testX.shape)
            print("bal sub_testy", testy.shape)
            
            run_training(trainX, trainy, testX, testy)

           


        

run()
