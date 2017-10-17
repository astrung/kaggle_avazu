print(__doc__)
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_selection import RFE


def split_X_and_Y(data):
    Y_data = data[['click']]
    del data['click']
    X_data = data

    return X_data, Y_data


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


data = pd.read_table("C:\\Users\\tuana\\Desktop\\kaggle\\sample_data.csv", sep=',', index_col=False,
                     nrows=10000,
                     usecols=range(1, 24))
original_data = data.copy()
list_hash_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                     'device_model',
                     'device_id', 'device_ip']
list_number_hashing_features = [5]

if __name__ == "__main__":
    for number_hashing_features in list_number_hashing_features:
        print("Number of hashing features : %d" % number_hashing_features)
        data = original_data
        hasher = FeatureHasher(n_features=number_hashing_features, input_type='string')

        for column_name in list_hash_columns:
            data = hashing_data(data, column_name, hasher, number_hashing_features)

        data = extract_time_stamp_feature(data)
        X, y = split_X_and_Y(data)
        sgd = SGDClassifier(n_jobs=-1, tol=0.00001, loss="log", alpha=0.0001,learning_rate='invscaling',eta0=0.005)
        rfecv = RFECV(estimator=sgd, step=1, cv=10,
                      scoring='neg_log_loss', n_jobs=3)

        rfecv.fit(X, y['click'])

        print("Optimal number of features : %d" % rfecv.n_features_)

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        joblib.dump(rfecv, 'reduced_model.pkl')