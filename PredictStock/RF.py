import numpy as np
import joblib
import os
import time
import pandas
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,\
    balanced_accuracy_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler


def train():

    result_name = "Results/RF_optimized.joblib"
    data_x = 'Data/x_new.csv'
    data_y = 'Data/y_new.csv'

    X = np.genfromtxt(data_x, delimiter=',', skip_header=1)
    Y = np.genfromtxt(data_y, delimiter=',', skip_header=1)

    print("Data is loaded")

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    # print(f'size X_train: {X_train.size}')
    print(f'size Y_test: {Y_test.size}')

    # Undersampling to get a similar distributed classes
    rus = RandomUnderSampler()
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    print(f'size Y_RUS: {Y_resampled.size}')

    t = pd.Series(Y_resampled)
    print(t.value_counts())

    # Default settings
    #print("Default settings")
    #rf = RandomForestClassifier(n_jobs=-1, verbose=2)

    # Hyperparameter optimized settings
    #print("Hyperparameter optimal settings")
    rf = RandomForestClassifier(n_estimators=68, min_samples_split=5, min_samples_leaf=9, max_features="sqrt",
                                max_depth=10, criterion="entropy", bootstrap=True, verbose=2)

    rf.fit(X_resampled, Y_resampled)
    print("Fit finished")

    # Get the predictions
    predictions_test = rf.predict(X_test)
    print("Predictions are made")

    print(confusion_matrix(Y_test, predictions_test))
    print('\n')
    print(classification_report(Y_test, predictions_test))
    print('\n')
    print(f'MCC_test: {matthews_corrcoef(Y_test, predictions_test)}')
    print('\n')
    print(f'Balanced Accuracy: {balanced_accuracy_score(Y_test, predictions_test)}')

    # safe the model
    joblib.dump(rf, result_name, compress=3)

    os.system("say 'finish'")


def HP_opt():
    start_proc = time.time()

    # Load data
    data_x = 'Data/x_new.csv'
    data_y = 'Data/y_new.csv'

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
    model = RandomForestClassifier()

    # define search space
    rf_params_rand = {"n_estimators": [int(x) for x in np.linspace(1, 100, num=20)],
                      "max_features": ['sqrt', 'log2'],
                      "max_depth": [int(x) for x in np.linspace(10, 100, num=10)],
                      "min_samples_leaf": [int(x) for x in np.linspace(2, 10, num=9)],
                      "min_samples_split": [int(x) for x in np.linspace(1, 5, num=5)],
                      "criterion": ['gini', 'entropy'],
                      "bootstrap": [True, False]}

    # define search, cv = 5 meaning stratisfied 5-fold cross-validation
    search = RandomizedSearchCV(estimator=model, param_distributions=rf_params_rand,
                                scoring='accuracy', cv=5, refit=True, n_iter=200, verbose=2, n_jobs=-1)
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




def predict(x, y):
    loaded_rf = joblib.load("/Users/benedictrau/Documents/GitHub/Masterarbeit/PredictStock/Results/RF_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    prediction = int(loaded_rf.predict(z))
    return prediction



def class_probability(x, y):
    loaded_rf = joblib.load("/Users/benedictrau/Documents/GitHub/Masterarbeit/PredictStock/Results/RF_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    class_probability = loaded_rf.predict_proba(z)
    return class_probability

#print(predict(10, 10))