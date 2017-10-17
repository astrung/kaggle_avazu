import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
import time
from sklearn.externals import joblib


def split_X_and_Y(training, test):
    Y_train = training[['click']]
    del training['click']
    X_train = training

    Y_test = test[['click']]
    del test['click']
    X_test = test
    return X_train, Y_train, X_test, Y_test

def extract_time_stamp_feature(data_frame):
    data_frame['dow'] = data_frame['hour'].apply(lambda x: (x % 10000 / 100) % 7)  # day of week
    data_frame['hour'] = data_frame['hour'].apply(lambda x: x % 100)
    return data_frame

def hashing_data(data_frame, column_name, hasher, number_of_features):
    list_column_name = []
    for i in range(number_of_features):
        list_column_name.append(column_name + str(i))
    new_set_of_columns = hasher.transform(data_frame[column_name].values)
    test = pd.DataFrame(new_set_of_columns.toarray(), columns=list_column_name)
    result = pd.concat([data_frame, test], axis=1)
    del result[column_name]
    return result

def get_predict(test_data, hasher, list_hash_columns,number_hash_feature):
    Y_id = test_data[['id']]
    del test_data['id']
    test_data = test_data
    for column_name in list_hash_columns:
        test_data = hashing_data(test_data, column_name, hasher,number_hash_feature)
    test_data = extract_time_stamp_feature(test_data)
    raw_result = sgd.predict_proba(test_data)

    predict_frame = pd.DataFrame(raw_result, columns=['non_click', 'click'])
    del predict_frame['non_click']
    result = pd.concat([Y_id, predict_frame], axis=1)
    return result

def write_predict_csv(write_index, data, path, hasher, list_hash_columns,number_hash_feature):
    data = data.reset_index(drop=True)
    new_scaler_data = get_predict(data, hasher, list_hash_columns,number_hash_feature)
    if write_index == 0:
        new_scaler_data.to_csv(path, mode='a', index=False
                               # ,float_format='string'
                               )
    else:
        new_scaler_data.to_csv(path, mode='a', header=False,
                               index=False
                               # ,float_format='string'
                               )
    time.sleep(1)

all_classes = np.array([0, 1])
batch_size = 100000
list_hash_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                     'device_model',
                     'device_id', 'device_ip']

chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\scale_train.csv', chunksize=batch_size, sep=','
                       , usecols=range(1, 24), index_col=False)
split_data = True
global_test_set = None
sgd = SGDClassifier(n_jobs=-1, tol=0.00001, loss="log",alpha=0.001)
number_of_hashing_feature = 5
hasher = FeatureHasher(n_features=number_of_hashing_feature, input_type='string')
index = 0

for data in chunks:
    if split_data:
        training, test = train_test_split(data, shuffle=True, train_size=0.5)
        training = data
        global_test_set = test.copy()
        split_data = False
    else:
        training = data
        test = global_test_set

    training = training.reset_index(drop=True)
    test = test.reset_index(drop=True)

    for column_name in list_hash_columns:
        training = hashing_data(training, column_name, hasher, number_of_hashing_feature)
        test = hashing_data(test, column_name, hasher, number_of_hashing_feature)

    training = extract_time_stamp_feature(training)
    test = extract_time_stamp_feature(test)

    X_train, Y_train, X_test, Y_test = split_X_and_Y(training, test)

    sgd.partial_fit(X_train, Y_train.values.ravel(), classes=all_classes)
    score = sgd.score(X_test, Y_test.values.ravel())
    print("sgd : %f " % score)

joblib.dump(sgd, "C:\\Users\\tuana\\PycharmProjects\\kaggle_avazu\\logistic_regression\\sgd_model.pkl")
# sgd = joblib.load('model_submit.pkl')
test_chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\scale_test.csv', chunksize=batch_size, sep=',',
                            index_col=False, dtype={'id': str})
write_test_csv_index = 0
for test_data in test_chunks:
    write_predict_csv(write_test_csv_index, test_data, "C:\\Users\\tuana\\Desktop\\kaggle\\result2.csv", hasher,
                      list_hash_columns,number_of_hashing_feature)

    write_test_csv_index = write_test_csv_index + 1
