import pickle

import numpy as np
import theano

from utils import load_gz, toFang


def save_data(data_path):
    ## Load training data
    TRAIN_PATH = data_path + 'cullpdb+profile_6133_filtered.npy.gz'
    X_in = load_gz(TRAIN_PATH)
    X_train = np.reshape(X_in, (5534, 700, 57))
    del X_in
    X_train = X_train[:, :, :]

    ## Load test data
    TEST_PATH = data_path + 'cb513+profile_split1.npy.gz'
    X_test_in = load_gz(TEST_PATH)
    X_test = np.reshape(X_test_in, (514, 700, 57))
    del X_test_in
    X_test = X_test[:, :, :]

    ## Stack
    X = np.vstack((X_train, X_test))
    labels = X[:, :, 22:30]
    mask = X[:, :, 30] * -1 + 1

    a = np.arange(0, 21)
    b = np.arange(35, 56)
    c = np.hstack((a, b))
    X = X[:, :, c]

    # getting meta
    num_seqs = np.size(X, 0)
    seqlen = np.size(X, 1)

    ## REMAKING LABELS
    X = X.astype(theano.config.floatX)
    mask = mask.astype(theano.config.floatX)
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seqs, seqlen))
    for i in range(np.size(labels, axis=0)):
        labels_new[i, :] = np.dot(labels[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels = labels_new

    print("X.shape", X.shape)
    print("labels.shape", labels.shape)

    X_am, X_psmm = toFang(X, mask)

    with open(data_path + "data.pkl", "wb") as f:
        pickle.dump((X_am, X_psmm, mask, labels), f, protocol=2)
    print("Data saved")


def load_data(data_path, first_seq=0, num_seqs=0):
    with open(data_path + "data.pkl", "rb") as f:

        X_am, X_psmm, mask, labels = pickle.load(f)

        if num_seqs == 0:
            last_seq = len(X_am)
        else:
            last_seq = first_seq + num_seqs

        return X_am[first_seq:last_seq], X_psmm[first_seq:last_seq], mask[first_seq:last_seq], labels[
                                                                                               first_seq:last_seq]

def save_predictions():
    X_am, X_pssm, mask, labels_test = load_data("")
    print("Data loaded")

    ## Load model
    model = load_model("modelQ8.h5")
    print("Model loaded")

    ## Make predictions
    predictions = model.predict([X_am, X_pssm])
    print(predictions.shape)
    print("Predictions made")

    with open("predictions.pkl", 'wb') as f:
        pickle.dump(predictions, f, protocol=2)


if __name__ == "__main__":
    # save_predictions()
    # save_data("")
    pass
