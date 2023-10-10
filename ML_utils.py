import numpy as np
import sys

# balance data
def balance_data(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    bal_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    X_train_bal = np.empty((0, X.shape[1]))
    X_test_bal = np.empty((0, X.shape[1]))
    y_train_bal = np.empty((0, y.shape[1]))
    y_test_bal = np.empty((0, y.shape[1]))
    print("X:", X.shape)
    print("y:", y.shape)
    for idx, item in enumerate(X):
        bal_val = 1356 
        if sys.argv[4] == 'split':
            bal_val = 40
        if bal_dict[int(np.argmax(y[idx]))] <= bal_val:
            X_train_bal = np.concatenate((X_train_bal, [item]))
            y_train_bal = np.concatenate((y_train_bal, [y[idx]]))
            bal_dict[int(np.argmax(y[idx]))] += 1
        else:
            X_test_bal = np.concatenate((X_test_bal, [item]))
            y_test_bal = np.concatenate((y_test_bal, [y[idx]]))

        print("\rBalancing data progress: " + str(idx + 1) + "/" + str(len(X)), end="")
    print(bal_dict)
    return X_train_bal, y_train_bal, X_test_bal, y_test_bal
