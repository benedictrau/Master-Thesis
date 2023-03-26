import numpy as np
import joblib
import os

from sklearn.metrics import classification_report, accuracy_score, \
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import time

### ABSOLUTER PFAD MUSS ANGEPASST WERDEN IN DEN BEIDEN LETZTEN FUNKTIONEN ###

def train():
    start_proc = time.time()

    result_name = "Results/XGB_optimized.joblib"
    data_x = 'Data/x_new.csv'
    data_y = 'Data/y_new.csv'

    X = np.genfromtxt(data_x, delimiter=',', skip_header=1)
    Y = np.genfromtxt(data_y, delimiter=',', skip_header=1)

    print("Data is loaded")

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

    # Undersampling to get a similiar distributed classes
    rus = RandomUnderSampler(random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)

    # Default settings
    #print("Default settings")
    #xgb_model = xgb.XGBClassifier()

    # Hyperparameter optimized settings
    print("Hyperparameter optimal settings")
    xgb_model = xgb.XGBClassifier(subsample=0.6, n_estimators=80, max_depth=50, learning_rate=0.55,
                                  colssample_bytree=0.4, colsample_bylevel=0.9)
    #xgb_model = xgb.XGBClassifier(subsample=0.8, n_estimators=1000, max_depth=35, learning_rate=0.55,
    #                              colssample_bytree=0.7, colsample_bylevel=0.4)

    xgb_model.fit(X_resampled, Y_resampled)
    print("Fit finished")

    # Get the predictions
    predictions_test = xgb_model.predict(X_test)
    print("Predictions are made")

    print(confusion_matrix(Y_test, predictions_test))
    print('\n')
    print(classification_report(Y_test, predictions_test))
    print('\n')
    print(f'MCC_test: {matthews_corrcoef(Y_test, predictions_test)}')
    print('\n')
    print(f'Balanced Accuracy: {balanced_accuracy_score(Y_test, predictions_test)}')

    # safe the model
    joblib.dump(xgb_model, result_name, compress=3)

    end_proc = time.time()
    print('Required time: {:5.3f}s'.format(end_proc - start_proc))

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
    model = xgb.XGBClassifier()

    # define search space
    xgb_params_rand = {'max_depth': [int(x) for x in np.linspace(5, 50, num=10)],
                       'learning_rate': np.arange(0.05, 0.95, 0.1),
                       'subsample': np.arange(0.5, 1.0, 0.1),
                       'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                       'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
                       'n_estimators': [int(x) for x in np.linspace(5, 100, num=20)]}

    # define search, cv = 5 meaning stratisfied 5-fold cross-validation
    search = RandomizedSearchCV(estimator=model, param_distributions=xgb_params_rand,
                                scoring='accuracy', cv=5, refit=True, n_iter=50, verbose=1, n_jobs=-1)

    # execute search
    result = search.fit(X_resampled, Y_resampled)
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

    ## ABSOLUTE PATH ##
    loaded_rf = joblib.load("/Users/benedictrau/Documents/GitHub/Master-Thesis/PredictStock/Results/XGB_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    prediction = int(loaded_rf.predict(z))
    #print(f'prediction: {prediction}')
    return prediction



def class_probability(x, y):

    ## ABSOLUTE PATH ##
    loaded_rf = joblib.load("/Users/benedictrau/Documents/GitHub/Master-Thesis/PredictStock/Results/XGB_optimized.joblib")
    z = np.array([[x, y]], dtype=object)
    class_probability = loaded_rf.predict_proba(z)
    #print(f'class_probability: {class_probability}')
    return class_probability

#test = class_probability(52, 5)
#print(test)

#train()
#HP_opt()

