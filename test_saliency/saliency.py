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
        for am, code in aaMap_fang.items():
            if code == number and am is not 'NoSeq':
                decoded_seq.append(am)

    return "".join(decoded_seq)


def toFang(X, mask):
    # Permutation from Troyanska's pssm arranging (ACDEFGHIKLMNPQRSTVWXY) to Fang's ('ARNDCQEGHILKMFPSTWYV NoSeq')
    sorted_fang = sorted(aaMap_fang.keys(), key=lambda letter: aaMap_fang[letter])

    aaMap = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    pssm = 'ACDEFGHIKLMNPQRSTVWXY'
    aaMap_jurtz = {amino: i for i, amino in enumerate(aaMap)}
    pssm_jurtz = {amino: i for i, amino in enumerate(pssm)}

    index_am = np.array([aaMap_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])
    index_pssm = np.array([pssm_jurtz[letter] for letter in sorted_fang if letter is not 'NoSeq'])

    X_am = X[:, :, index_am]
    X_pssm = X[:, :, index_pssm + 21]

    # Add NoSeq class
    X_am = np.concatenate([X_am, mask[:, :, None]], axis=2)
    X_pssm = np.concatenate([X_pssm, mask[:, :, None]], axis=2)

    return X_am, X_pssm


def convertPredictQ8Result2HumanReadable(predictedSS):
    predSS = np.argmax(predictedSS, axis=-1)
    ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
    result = []
    for i in range(0, 700):
        result.append(ssConvertMap[predSS[i]])
    return ''.join(result)


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


def compute_tensor_saliency(X_am, X_pssm, first_seq, num_seqs):
    if X_am.ndim == 2:
        X_am = X_am[None, ...]
        X_pssm = X_pssm[None, ...]

    model = load_model("modelQ8.h5")

    gradients = jacobian(model.outputs[0][:, :, 5].flatten(),
                         wrt=[model.inputs[0], model.inputs[1]])
    get_gradients = K.function(inputs=[model.inputs[0], model.inputs[1], K.learning_phase()],
                               outputs=gradients)
    grads = get_gradients([X_am, X_pssm, 0])

    with open(("saliencies"+str(first_seq)+"-"+str(first_seq+num_seqs)+".pkl"), 'wb') as f:
        pickle.dump(grads, f, protocol=2)


def compute_saliency(X_am, X_pssm, labels):
    num_seqs = np.size(X_am, 0)
    seqlen = np.size(X_am, 1)

    # 29 aminoacids per side
    window = 29

    ## Load model
    model = load_model("modelQ8.h5")

    ## Make predictions
    predictions = model.predict([X_am, X_pssm])
    print(predictions.shape)

    start_time = time.time()
    prev_time = start_time

    saliencies = np.zeros((2, num_seqs, seqlen, seqlen, 21))
    saliency_info = pd.DataFrame(
        columns=["Seq", "Pos", "Class", "Prediction", "Aminoacids", "Predictions", "True labels"])

    ssConvertMap = {0: 'C', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'H', 6: 'S', 7: 'T', 8: ''}
    for seq in range(num_seqs):

        gato = decode(X_am[seq])
        perro = convertPredictQ8Result2HumanReadable(predictions[seq])
        conejo = "".join([ssConvertMap[el] for el in labels[seq]])

        for pos in range(seqlen):
            if labels[seq, pos] == np.argmax(predictions[seq, pos]):
                new_row = len(saliency_info)

                target_class = labels[seq, pos]

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
                saliency_info.loc[new_row, "Predictions"] = perro[init:pos] + " " + perro[pos] + " " + perro[
                                                                                                       pos + 1:end]
                saliency_info.loc[new_row, "True labels"] = conejo[init:pos] + " " + conejo[pos] + " " + conejo[
                                                                                                         pos + 1:end]

                ## Compute gradients
                gradients = theano.gradient.grad(model.outputs[0][seq, pos, target_class],
                                                 wrt=[model.inputs[0], model.inputs[1]])
                get_gradients = K.function(inputs=[model.inputs[0], model.inputs[1], K.learning_phase()],
                                           outputs=gradients)
                inputs = [X_am[seq, ...], X_pssm[seq, ...], 0]
                inputs = [inputs[0][None, ...], inputs[1][None, ...], inputs[2]]
                grads = get_gradients(inputs)
                grads = np.array(grads)
                saliencies[:, seq, pos, :, :] = grads[:, 0, ...]

        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        est_time_left = time_since_start * num_seqs
        eta = datetime.now() + timedelta(seconds=est_time_left)
        eta_str = eta.strftime("%c")
        print("  %s since start (%.2f s)" % (time_since_start, time_since_prev))
        print("  estimated %s to go (ETA: %s)" % (est_time_left, eta_str))

    with open(("saliencies.pkl"), 'wb') as f:
        pickle.dump((saliencies, saliency_info), f, protocol=2)


def main_saliencies():
    first_seq=1
    num_seqs=20
    X_am, X_pssm, mask, labels = load_data("", first_seq=first_seq, num_seqs=num_seqs)
    # compute_saliency(X_am, X_pssm, labels)
    compute_tensor_saliency(X_am, X_pssm, first_seq, num_seqs)


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


def jacobian(expression, wrt, consider_constant=None,
             disconnected_inputs='raise'):
    """
    Compute the full Jacobian, row by row.
    Parameters
    ----------
    expression : Vector (1-dimensional) :class:`~theano.gof.graph.Variable`
        Values that we are differentiating (that we want the Jacobian of)
    wrt : :class:`~theano.gof.graph.Variable` or list of Variables
        Term[s] with respect to which we compute the Jacobian
    consider_constant : list of variables
        Expressions not to backpropagate through
    disconnected_inputs: string
        Defines the behaviour if some of the variables
        in `wrt` are not part of the computational graph computing `cost`
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.
    Returns
    -------
    :class:`~theano.gof.graph.Variable` or list/tuple of Variables (depending upon `wrt`)
        The Jacobian of `expression` with respect to (elements of) `wrt`.
        If an element of `wrt` is not differentiable with respect to the
        output, then a zero variable is returned. The return value is
        of same type as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    from theano.tensor import arange
    # Check inputs have the right format
    assert isinstance(expression, theano.gof.Variable), \
        "tensor.jacobian expects a Variable as `expression`"
    assert expression.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expression`. If not use flatten to make it a vector")

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    if expression.ndim == 0:
        # expression is just a scalar, use grad
        return theano.gradient.format_as(using_list, using_tuple,
                                         theano.gradient.grad(expression,
                                                              wrt,
                                                              consider_constant=consider_constant,
                                                              disconnected_inputs=disconnected_inputs))

    def inner_function(*args):
        idx = args[0]
        expr = args[1]
        rvals = []
        for inp in args[2:]:
            rval = theano.gradient.grad(expr[idx],
                                        inp,
                                        consider_constant=consider_constant,
                                        disconnected_inputs=disconnected_inputs)
            rvals.append(rval)
        return rvals

    # Computing the gradients does not affect the random seeds on any random
    # generator used n expression (because during computing gradients we are
    # just backtracking over old values. (rp Jan 2012 - if anyone has a
    # counter example please show me)
    jacobs, updates = theano.scan(inner_function,
                                  sequences=arange(expression.shape[0]),
                                  non_sequences=[expression] + wrt)
    # assert not updates, \
    #     ("Scan has returned a list of updates. This should not "
    #      "happen! Report this to theano-users (also include the "
    #      "script that generated the error)")

    return theano.gradient.format_as(using_list, using_tuple, jacobs)


if __name__ == "__main__":
    import sys

    with open("job_output.txt", "w") as f:
        sys.stdout = f
        main_saliencies()
        # save_predictions()
        # save_data("")
