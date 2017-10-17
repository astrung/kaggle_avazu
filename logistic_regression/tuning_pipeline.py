from __future__ import print_function
from sklearn.feature_extraction import FeatureHasher
from pprint import pprint
from time import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def split_X_and_Y(data):
    Y_data = data[['click']]
    del data['click']
    X_data = data

    return X_data, Y_data


def extract_time_stamp_feature(data_frame):
    data_frame['dow'] = data_frame['hour'].apply(lambda x: (x % 10000 / 100) % 7)  # day of week
    data_frame['hour'] = data_frame['hour'].apply(lambda x: x % 100)
    return data_frame


def hashing_data(data_frame, column_name, hasher,number_of_features):
    list_column_name = []
    for i in range(number_of_features):
        list_column_name.append(column_name + str(i))
    new_set_of_columns = hasher.transform(data_frame[column_name].values)
    test = pd.DataFrame(new_set_of_columns.toarray(), columns=list_column_name)
    result = pd.concat([data_frame, test], axis=1)
    del result[column_name]
    return result


if __name__ == "__main__":
    data = pd.read_table("C:\\Users\\tuana\\Desktop\\kaggle\\sample_data.csv", sep=',', index_col=False,usecols=range(1, 24))
    original_data=data.copy()
    list_hash_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                         'device_model',
                         'device_id', 'device_ip']
    list_number_hashing_features=[5]
    pipeline = Pipeline([
        ('clf', SGDClassifier(tol=0.000001,loss="log",max_iter=10000)),
    ])

    parameters = {
        'clf__eta0': (0.005,0.001, 0.0005 ),
        'clf__alpha': (0.01,0.001,0.0001),
        'clf__learning_rate':(['optimal','invscaling']),
    }
    log_loss_build = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, cv=10,scoring='neg_log_loss')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    for number_hashing_features in list_number_hashing_features:
        print("Number of hashing features : %d" % number_hashing_features)
        data=original_data
        reduced_model = joblib.load(
            "C:\\Users\\tuana\\PycharmProjects\\kaggle_avazu\\logistic_regression\\reduced_model.pkl")

        hasher = FeatureHasher(n_features=number_hashing_features, input_type='string')

        for column_name in list_hash_columns:
            data = hashing_data(data, column_name, hasher,number_hashing_features)

        data = extract_time_stamp_feature(data)

        X_data, Y_data = split_X_and_Y(data)

        grid_search.fit(X_data, Y_data['click'])

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        print(grid_search.grid_scores_)
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
