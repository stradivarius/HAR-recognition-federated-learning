import numpy as np
import pandas as pd


def get_anovaf(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    x_tmp = np.zeros(X.shape[0])
    x_medio = 0

    # inserisco in x_tmp[i] la somma dei valori delle feature con indice i
    for idx, val in enumerate(X):
        for i, v in enumerate(val):
            x_tmp[i] += v
    # per ogni sommatoria di feature in x_tmp calcolo il valore medio della feature
    # dividendo la sommatoria per il numero di valori di ogni feature
    for idx, val in enumerate(x_tmp):
        x_tmp[idx] /= X.shape[0]
    df = pd.DataFrame(X)
    df = df.T
    anova_prog_max = ((X.shape[1] * (X.shape[0] + 6)) * 2) + ((X.shape[1] * 6) * 2)
    anova_act_prog = 0
    elem_count = 0
    x_medio_lst = []
    x_medio_per_class = []
    
    # ogni j è una riga del dataframe 
    # composta da 10299 elementi indicizzati da 0 a 10298
    # e ogni riga è la raccolta di valori di ogni feature
    for j in df.iterrows():
        x_medio_tmp = 0
        elem_count_tmp = 0
        class_medio = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            # per ogni classe conta il numero di feature di quella classe
            # perchè += 1 e non += j[1][i] ?
            class_medio[np.argmax(y[i])] += 1
            # calcola la somma dei valori di ogni feature
            x_medio += j[1][i]
            x_medio_tmp += j[1][i]
            elem_count += 1
            elem_count_tmp += 1
            anova_act_prog += 1
            # print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        for i in range(len(class_medio)):
            # print(elem_count_tmp)
            class_medio[i] /= elem_count_tmp
            anova_act_prog += 1
        print(
            "\rAnova progress: ",
            round(((anova_act_prog / anova_prog_max) * 100), 2),
            "%",
            end="",
        )
        # contiene quanti elementi ha una feature in ogni classe
        x_medio_per_class.append(class_medio)
        # il valore x_medio per feature tra le classi
        x_medio_lst.append(x_medio_tmp / elem_count_tmp)
    x_medio = x_medio / elem_count
    varianza_t_avgot = 0
    varianza_par_lst = []
    varianza_per_classe = []
    row_count = 0
    # per ogni feature con i suoi valori in tutte le classi
    for j in df.iterrows():
        varianza_par = 0
        elem_count_tmp = 0
        class_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            #per ogni classe calcola la sommatorie della differenza tra valore e valore medio, il tutto alla seconda
            # perchè usa x_medio_per class che contiene per ogni classe il numero di elementi della feature
            # e non la sommatoria dei valori di x per ogni classe
            class_dict[np.argmax(y[i])] += pow(
                j[1][i] - x_medio_per_class[row_count][np.argmax(y[i])], 2
            )
            # calcola la sommatoria delle differenze globalmente, ovvero tra classi diverse
            varianza_t_avgot += pow(j[1][i] - x_medio, 2)
            varianza_par += pow(j[1][i] - x_medio, 2)
            elem_count_tmp += 1
            anova_act_prog += 1
            # print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
        # calcola la varianza per ogni classe
        for i in range(len(class_dict)):
            class_dict[i] /= elem_count_tmp
            anova_act_prog += 1
        print(
            "\rAnova progress: ",
            round(((anova_act_prog / anova_prog_max) * 100), 2),
            "%",
            end="",
        )
        varianza_per_classe.append(class_dict)
        varianza_par_lst.append(varianza_par / elem_count_tmp)
    # calcola la varianza tra tutte le classi
    varianza_t_avgot /= elem_count
    varianza_media_classi = []
    varianza_min_classi = []
    classe_varianza_min = []
    classi = 0
    # per ogni index delle feature 
    for i in range(len(varianza_per_classe)):
        accumulatore = 0
        min_var_class = 1.0
        classe_min = 0
        # per ogni classe associata alla feature con index i
        for j in range(len(varianza_per_classe[i])):
            classi = len(varianza_per_classe[i])
            # somma la varianza di tutte le classi j associate alla feature i 
            accumulatore += varianza_per_classe[i][j]
            if varianza_per_classe[i][j] < min_var_class:
                min_var_class = varianza_per_classe[i][j]
                classe_min = j
            anova_act_prog += 1
        print(
            "\rAnova progress: ",
            round(((anova_act_prog / anova_prog_max) * 100), 2),
            "%",
            end="",
        )
        varianza_media_classi.append(accumulatore / classi)
        varianza_min_classi.append(min_var_class)
        classe_varianza_min.append(classe_min)
        # questa ulteriore divsione non mi è chiara
    for i in range(len(varianza_par_lst)):
        varianza_media_classi[i] /= varianza_par_lst[i]
        varianza_min_classi[i] /= varianza_par_lst[i]
        for j in range(len(varianza_per_classe[i])):
            varianza_per_classe[i][j] /= varianza_par_lst[i]
            anova_act_prog += 1
        print(
            "\rAnova progress: ",
            round(((anova_act_prog / anova_prog_max) * 100), 2),
            "%",
            end="",
        )
    print("\n")
    # rispettivamente contengono per ogni feature,  la media  tra le  varianze (valori  anova)
    #   e l’altra  che  contiene  il  minimo tra  levarianze.
    return varianza_media_classi, varianza_min_classi
