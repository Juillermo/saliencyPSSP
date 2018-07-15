# Install libraries: theano, keras
# Bring data files
# Bring model file

import pickle
import numpy as np
import theano
import keras.backend as K
from keras.models import load_model

from saliency import load_gz, toFang, decode, convertPredictQ8Result2HumanReadable


def load_data():
    ## Load training data
    TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
    X_in = load_gz(TRAIN_PATH)
    X_train = np.reshape(X_in, (5534, 700, 57))
    del X_in
    X_train = X_train[:2, :, :]

    ## Load test data
    TEST_PATH = 'cb513+profile_split1.npy.gz'
    X_test_in = load_gz(TEST_PATH)
    X_test = np.reshape(X_test_in, (514, 700, 57))
    del X_test_in
    X_test = X_test[:2, :, :]

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
    for i in range(np.size(labels, axis=0)):
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
    model = load_model("modelQ8.h5")

    ## Make predictions
    predictions = model.predict([X_test_am, X_test_pssm])
    print(predictions.shape)

    ## Compute saliencies
    saliencies = np.empty((1, 1), dtype=object)

    ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
    for seq in range(num_seqs):
        gato = decode(X_test_am[seq])
        perro = convertPredictQ8Result2HumanReadable(predictions[seq])
        conejo = "".join([ssConvertMap[el] for el in labels_test[seq]])

        for pos in range(seqlen):
            if labels_test[seq, pos] == np.argmax(predictions[seq,pos]):
                target_class = labels_test[seq, pos]
                saliency_object = {}

                ## Compute string aminoacids and predictions
                saliency_object["Class"] = ssConvertMap[target_class]
                saliency_object["Prediction"] = predictions[seq, pos, target_class]

                if pos >= window:
                    init = pos - window
                else:
                    init = 0

                if pos + window >= seqlen:
                    end = seqlen
                else:
                    end = pos + window + 1

                saliency_object["Aminoacids"] = gato[init: pos] + " " + gato[pos] + " " + gato[pos + 1: end]
                saliency_object["Predictions"] = perro[init:pos] + " " + perro[pos] + " " + perro[pos + 1:end]
                saliency_object["True labels"] = conejo[init:pos] + " " + conejo[pos] + " " + conejo[pos + 1:end]

                ## Compute gradients
                gradients = theano.gradient.grad(model.outputs[0][seq, pos, target_class],
                                                 wrt=[model.inputs[0], model.inputs[1]])
                get_gradients = K.function(inputs=[model.inputs[0], model.inputs[1], K.learning_phase()],
                                           outputs=gradients)
                inputs = [X_test_am[seq, ...], X_test_pssm[seq, ...], 0]
                inputs = [inputs[0][None, ...], inputs[1][None, ...], inputs[2]]
                grads = get_gradients(inputs)
                saliency_object["grads"] = np.array(grads)

                saliencies[seq, pos] = saliency_object

                print(seq, pos)

                return saliencies

if __name__ == "__main__":
    ## Load data
    X, mask_test, labels_test, num_seqs = load_data()
    X_am, X_pssm = toFang(X, mask_test)

    ## Compute saliencies
    saliencies = compute_saliency(X_am, X_pssm, labels_test)

    ## Save file
    with open(("saliencies.pkl"), 'wb') as f:
        pickle.dump(saliencies, f, pickle.HIGHEST_PROTOCOL)
