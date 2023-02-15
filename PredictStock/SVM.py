import numpy as np
import joblib
import os

from sklearn.metrics import classification_report, confusion_matrix, \
    matthews_corrcoef, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC as svc
import time


def train():
    result_name = "Results/SVM_optimized.joblib"
    data_x = 'Data/x_new_SVM.csv'
    data_y = 'Data/y_new_SVM.csv'

    X = np.genfromtxt(data_x, delimiter=',', skip_header=1)
    Y = np.genfromtxt(data_y, delimiter=',', skip_header=1)

    print("Data is loaded")

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

    # Undersampling to get a similiar distributed classes
    rus = RandomUnderSampler()
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)

    # Default settings
    print("Default settings")
    svm = svc(probability=True, random_state=1)

    # Hyperparameter optimized settings
    #print("Hyperparameter optimal settings")
    #svm = svc(probability=True, C=2, gamma=0.3, kernel="rbf", degree=1, verbose=2)

    svm.fit(X_resampled, Y_resampled)
    print("Fit finished")

    predictions_test = svm.predict(X_test)
    print("Predictions are made")

    print(confusion_matrix(Y_test, predictions_test))
    print('\n')
    print(classification_report(Y_test, predictions_test))
    print('\n')
    print(f'MCC_test: {matthews_corrcoef(Y_test, predictions_test)}')
    print('\n')
    print(f'Balanced Accuracy: {balanced_accuracy_score(Y_test, predictions_test)}')

    # safe the model
    joblib.dump(svm, result_name, compress=3)

    os.system("say 'finish'")



def HP_opt():
    start_proc = time.time()

    # Load data
    data_x = 'Data/x_new_SVM.csv'
    data_y = 'Data/y_new_SVM.csv'

    X = np.genfromtxt(data_x, delimiter=',', skip_header=1)
    Y = np.genfromtxt(data_y, delimiter=',', skip_header=1)
    print("Data is loaded")

    # Apply train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    # print(f'size Y_train: {Y_train.size}')

    # Undersampling to get a similiar distributed classes
    rus = RandomUnderSampler(random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    # print(f'size Y_resampled: {Y_resampled.size}')

    # define the model
    svm = svc()

    # define search space
    svm_params_rand = {"C": np.arange(1, 102, 10),
                       "gamma": np.arange(0.1, 1, 0.2),
                       "kernel": ['rbf', 'poly', 'sigmoid'],
                       # "kernel": ['rbf', 'sigmoid'],
                       "degree": [1, 2, 3, 4]}

    # define search, cv = 5 meaning stratisfied 5-fold cross-validation
    search = RandomizedSearchCV(estimator=svm, param_distributions=svm_params_rand,
                                scoring='accuracy', cv=5, refit=True, n_iter=5, verbose=1, n_jobs=-1)
    # execute search
    search.fit(X_resampled, Y_resampled)
    # get the best performing model fit on the whole training set
    best_model = search.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_resampled)
    # evaluate the model
    acc = accuracy_score(Y_resampled, yhat)
    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, search.best_score_, search.best_params_))

    end_proc = time.time()
    print('Required time: {:5.3f}s'.format(end_proc - start_proc))


HP_opt()


def predict(x, y):
    loaded_svm = joblib.load(
        "/Users/benedictrau/Documents/GitHub/Masterarbeit/PredictStock/Results/SVM_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    prediction = int(loaded_svm.predict(z))
    return prediction




def class_probability(x, y):
    loaded_svm = joblib.load(
        "/Users/benedictrau/Documents/GitHub/Masterarbeit/PredictStock/Results/SVM_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    class_probability = loaded_svm.predict_proba(z)
    return class_probability