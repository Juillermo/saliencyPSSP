# Install libraries: theano, keras
# Bring data files
# Bring model file

import gzip
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import theano
import keras.backend as K
from keras.models import load_model


def load_gz(path):  # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


aaMap_fang = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12,
              'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'NoSeq': 20}


def decode(coded_seq):
    coded_seq = np.argmax(coded_seq, axis=1)
    decoded_seq = []

    for number in coded_seq:
        for am, code in aaMap_fang.iteritems():
            if code == number and am is not 'NoSeq':
                decoded_seq.append(am)

    return "".join(decoded_seq)


def toFang(X_test, mask_test):
    # Permutation from Troyanska's pssm arranging (ACDEFGHIKLMNPQRSTVWXY) to Fang's ('ARNDCQEGHILKMFPSTWYV NoSeq')
    sorted_fang = sorted(aaMap_fang.keys(), key=lambda letter: aaMap_fang[letter])

    aaMap = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    pssm = 'ACDEFGHIKLMNPQRSTVWXY'
    aaMap_jurtz = {amino: i for i, amino in enumerate(aaMap)}
    pssm_jurtz = {amino: i for i, amino in enumerate(pssm)}

    index_am = np.array([aaMap_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])
    index_pssm = np.array([pssm_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])

    X_test_am = X_test[:, :, index_am]
    X_test_pssm = X_test[:, :, index_pssm + 21]

    # Add NoSeq class
    X_test_am = np.concatenate([X_test_am, mask_test[:, :, None]], axis=2)
    X_test_pssm = np.concatenate([X_test_pssm, mask_test[:, :, None]], axis=2)

    return X_test_am, X_test_pssm


def convertPredictQ8Result2HumanReadable(predictedSS):
    predSS = np.argmax(predictedSS, axis=-1)
    ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
    result = []
    for i in range(0, 700):
        result.append(ssConvertMap[predSS[i]])
    return ''.join(result)


def load_data():
    ## Load training data
    TRAIN_PATH = '../secondary_proteins_prediction/data/cullpdb+profile_6133_filtered.npy.gz'
    X_in = load_gz(TRAIN_PATH)
    X_train = np.reshape(X_in, (5534, 700, 57))
    del X_in
    X_train = X_train[:, :, :]

    ## Load test data
    TEST_PATH = '../secondary_proteins_prediction/data/cb513+profile_split1.npy.gz'
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

    #### REMAKING LABELS ####
    X = X.astype(theano.config.floatX)
    mask = mask.astype(theano.config.floatX)
    vals = np.arange(0, 8)
    labels_new = np.zeros((num_seqs, seqlen))
    for i in xrange(np.size(labels, axis=0)):
        labels_new[i, :] = np.dot(labels[i, :, :], vals)
    labels_new = labels_new.astype('int32')
    labels = labels_new

    print("X.shape", X.shape)
    print("labels.shape", labels.shape)

    return X, mask, labels, num_seqs


def compute_saliency(X_test_am, X_test_pssm, labels_test):
    num_seqs = np.size(X_test_am, 0)
    seqlen = np.size(X_test_am, 1)

    # 29 aminoacids per side
    window = 29

    ## Load model
    model = load_model("../Standalone/data/modelQ8.h5")

    ## Make predictions
    predictions = model.predict([X_test_am, X_test_pssm])
    print(predictions.shape)


    start_time = time.time()
    prev_time = start_time

    saliencies = np.zeros((2, num_seqs, seqlen, 21))
    saliency_info = pd.DataFrame(columns=["Seq", "Pos", "Class", "Prediction", "Aminoacids", "Predictions", "True labels"])

    ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
    for seq in range(num_seqs):

        gato = decode(X_test_am[seq])
        perro = convertPredictQ8Result2HumanReadable(predictions[seq])
        conejo = "".join([ssConvertMap[el] for el in labels_test[seq]])

        for pos in range(seqlen):
            if labels_test[seq, pos] == np.argmax(predictions[seq, pos]):
                new_row = len(saliency_info)

                target_class = labels_test[seq, pos]

                ## Compute string aminoacids and predictions
                saliency_info.loc[new_row, "Class"] = ssConvertMap[target_class]
                saliency_info.loc[new_row, "Prediction"] = predictions[seq, pos, target_class]

                if pos >= window:
                    init = pos - window
                else:
                    init = 0

                if pos + window >= seqlen:
                    end = seqlen
                else:
                    end = pos + window + 1

                saliency_info.loc[new_row, "Aminoacids"] = gato[init: pos] + " " + gato[pos] + " " + gato[pos + 1: end]
                saliency_info.loc[new_row, "Predictions"] = perro[init:pos] + " " + perro[pos] + " " + perro[pos + 1:end]
                saliency_info.loc[new_row, "True labels"] = conejo[init:pos] + " " + conejo[pos] + " " + conejo[pos + 1:end]

                ## Compute gradients
                gradients = theano.gradient.grad(model.outputs[0][seq, pos, target_class],
                                                 wrt=[model.inputs[0], model.inputs[1]])
                get_gradients = K.function(inputs=[model.inputs[0], model.inputs[1], K.learning_phase()],
                                           outputs=gradients)
                inputs = [X_test_am[seq, ...], X_test_pssm[seq, ...], 0]
                inputs = [inputs[0][None, ...], inputs[1][None, ...], inputs[2]]
                grads = get_gradients(inputs)
                saliencies[:,seq,pos,:] = np.array(grads)

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * num_seqs
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print("  %s since start (%.2f s)" % (time_since_start, time_since_prev))
        print("  estimated %s to go (ETA: %s)" % (est_time_left, eta_str))

    return saliency_info, saliencies


def main():
    ## Load data
    X, mask, labels_test, num_seqs = load_data()
    X_test_am, X_test_pssm = toFang(X, mask)

    ## Compute saliencies
    saliencies, saliency_info = compute_saliency(X_test_am, X_test_pssm, labels_test)

    ## Save file
    with open(("saliencies.pkl"), 'w') as f:
        pickle.dump((saliencies, saliency_info), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
