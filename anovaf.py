import numpy as np
import pandas as pd


def get_anovaf(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    x_tmp = np.zeros(X.shape[0])
    x_medio = 0
    for idx, val in enumerate(X):
        for i, v in enumerate(val):
            x_tmp[i] += v
    for idx, val in enumerate(x_tmp):
        x_tmp[idx] /= X.shape[0]
    df = pd.DataFrame(X)
    df = df.T
    anova_prog_max = ((X.shape[1] * (X.shape[0] + 6)) * 2) + ((X.shape[1] * 6) * 2)
    anova_act_prog = 0
    elem_count = 0
    x_medio_lst = []
    x_medio_per_class = []
    for j in df.iterrows():
        x_medio_tmp = 0
        elem_count_tmp = 0
        class_medio = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            # dev_int += pow(j[1][i] - x_tmp[idx], 2)
            class_medio[np.argmax(y[i])] += 1
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
        x_medio_per_class.append(class_medio)
        x_medio_lst.append(x_medio_tmp / elem_count_tmp)
    x_medio = x_medio / elem_count
    varianza_t_avgot = 0
    varianza_par_lst = []
    varianza_per_classe = []
    row_count = 0
    for j in df.iterrows():
        varianza_par = 0
        elem_count_tmp = 0
        class_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for i in range(len(j[1])):
            class_dict[np.argmax(y[i])] += pow(
                j[1][i] - x_medio_per_class[row_count][np.argmax(y[i])], 2
            )
            varianza_t_avgot += pow(j[1][i] - x_medio, 2)
            varianza_par += pow(j[1][i] - x_medio, 2)
            elem_count_tmp += 1
            anova_act_prog += 1
            # print("\rAnova progress: ", round(((anova_act_prog / anova_prog_max) * 100), 2), "%", end="")
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
    varianza_t_avgot /= elem_count
    varianza_media_classi = []
    varianza_min_classi = []
    classe_varianza_min = []
    classi = 0
    for i in range(len(varianza_per_classe)):
        accumulatore = 0
        min_var_class = 1.0
        classe_min = 0
        for j in range(len(varianza_per_classe[i])):
            classi = len(varianza_per_classe[i])
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
    # rispettivamente contengono per ogni feature,  la media  tra le  varianze (valori  anova)
    #   e l’altra  che  contiene  il  minimo tra  levarianze.
    return varianza_media_classi, varianza_min_classi
