import pickle
import time
import argparse
import importlib
import glob
import os
import re

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import keras.backend as K
from keras.models import load_model
import lasagne as nn

from utils import decode, ssConvertMap, ssConvertString, convertPredictQ8Result2HumanReadable
from data_managing import load_data

batch_size = 64


def spot_best(X_am, X_pssm, labels):
    ## NEEDS REVISION

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

    with open(("saliencies.pkl"), 'wb') as f:
        pickle.dump((saliency_info), f, protocol=2)


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


def compute_tensor_jurtz(X_batch, mask_batch, batch, label, ini=0):
    """
    Computes the saliencies of a batch for a certain label. If ini!=0, it also repairs the ini sequence and
    computes the rest of the batch.

    Inputs:
        X_batch: batch of length 64, with its corresponding mask batch
        batch: batch number (absolute, for labeling)
        label: one of the 8 classes (see ssConvertString)
        ini: which sequence of the batch to start from. If not 0, it will also split the first sequence in 2

    Outputs:
        Saves individual saliencies in separated pickle files, properly labelled"""

    metadata_path = "dump_pureConv-20180804-010835-47.pkl"
    metadata = np.load(metadata_path)
    config_name = metadata['config_name']
    config = importlib.import_module("%s" % config_name)
    print("Using configurations: '%s'" % config_name)
    print("Build model")
    l_in, l_out = config.build_model()
    print("Build eval function")
    sym_x = T.tensor3()
    inference = nn.layers.get_output(
        l_out, sym_x, deterministic=True)
    print("Load parameters")
    nn.layers.set_all_param_values(l_out, metadata['param_values'])

    if ini != 0:
        print("Computing broken saliency")
        seq_len = int(np.sum(mask_batch[ini]))

        # FIRST HALF
        gradients = theano.gradient.jacobian(inference[ini, :seq_len // 2, ssConvertString.find(label)],
                                             wrt=sym_x)
        get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

        grads = np.array(get_gradients(X_batch))

        del get_gradients
        del gradients

        # SECOND HALF
        gradients = theano.gradient.jacobian(inference[ini, seq_len // 2:seq_len, ssConvertString.find(label)],
                                             wrt=sym_x)
        get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

        grads2 = np.array(get_gradients(X_batch))

        tot_grads = np.concatenate((grads[:, ini, seq_len], grads2[:, ini, seq_len]), axis=0)

        with open("saliencies_jurtz/saliencies{:4d}{:s}.pkl".format(batch_size * batch + ini, label),
                  'wb') as f:
            pickle.dump(tot_grads, f, protocol=2)

        ini += 1

    print("Compute saliencies")
    for batch_seq in range(ini, batch_size):
        seq_len = int(np.sum(mask_batch[batch_seq]))
        gradients = theano.gradient.jacobian(inference[batch_seq, :seq_len, ssConvertString.find(label)],
                                             wrt=sym_x)
        get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

        grads = get_gradients(X_batch)
        grads = np.array(grads)

        with open("saliencies_jurtz/saliencies{:4d}{:s}.pkl".format(batch_size * batch + batch_seq, label),
                  'wb') as f:
            pickle.dump(grads[:seq_len, batch_seq, :seq_len], f, protocol=2)
        print(batch_seq)


def main_saliencies_jurtz():
    parser = argparse.ArgumentParser(description='Compute saliencies (jurtz)')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='class to which gradients are computed (default H)')
    parser.add_argument('--batch', type=int, default=0, metavar='batch',
                        help='batch of which the gradient is calculated (default 0)')
    args = parser.parse_args()

    batch_size = 64
    if args.batch is not None:
        if args.batch < 5534 // batch_size:
            from data import get_train
            TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
            if args.batch < 5278 // batch_size:
                X, _, _, _, mask, _, _ = get_train(TRAIN_PATH)
                batch = args.batch
            else:
                _, X, _, _, _, mask, _ = get_train(TRAIN_PATH)
                batch = args.batch - 5278 // batch_size
        else:
            from data import get_test
            TEST_PATH = 'data/cb513+profile_split1.npy.gz'
            X, mask, _, _ = get_test(TEST_PATH)
            batch = args.batch - 5534 // batch_size

        idx = range(batch * batch_size, (args.batch + 1) * batch_size)
        compute_tensor_jurtz(X[idx], mask[idx], args.batch, args.label)


def calculate_SeqLogo(args):
    _, _, mask, _ = load_data("", num_seqs=args.num_seqs)
    del _
    window = 9

    total = np.zeros((2, 2 * window + 1, 21))  # one-hot/pssm, window-size, n aminoacids
    for seq in range(args.num_seqs):
        with open("saliencies/saliencies" + str(seq) + args.label + ".pkl", "rb") as f:
            saliencies = np.array(pickle.load(f))

        end_seq = int(sum(mask[seq]))
        for pos in range(end_seq):
            # Pre-window
            if pos > window:
                init = pos - window
                total[:, :window] += saliencies[:, pos, 0, init:pos, :]
            elif pos != 0:
                init = window - pos
                total[:, init:window] += saliencies[:, pos, 0, 0:pos, :]

            # Window
            total[:, window] = saliencies[:, pos, 0, pos, :]

            # Post-window
            if pos + window + 1 <= end_seq:
                end = pos + window + 1
                total[:, window + 1:] = saliencies[:, pos, 0, pos + 1:end, :]
            elif pos != end_seq:
                end = end_seq
                total[:, window + 1:-(pos + window + 1 - end)] = saliencies[:, pos, 0, pos + 1:end, :]

    with open("SeqLogo" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(total, f, protocol=2)


def main_SeqLogo():
    parser = argparse.ArgumentParser(description='Compute SeqLogo from saliencies')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='class to which gradients are computed (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of sequences aggregated for SeqLogo (default 2)')
    args = parser.parse_args()

    if args.seq is not None:
        calculate_SeqLogo(args)


def repare_saliencies():
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')
    files = glob.glob('saliencies*')
    exists = [False for _ in range(6018)]
    for el in files:
        num = int(re.search(r'\d+', el).group(0))
        exists[num] = True

    os.chdir(origin)
    from data import get_train, get_test
    TRAIN_PATH = 'cullpdb+profile_6133_filtered.npy.gz'
    TEST_PATH = 'cb513+profile_split1.npy.gz'
    X, _, _, _, mask, _, _ = get_train(TRAIN_PATH)

    def get_index(batch):
        return batch

    for batch in range(6018 // batch_size):
        if batch == 5278 // batch_size:
            _, X, _, _, _, mask, _ = get_train(TRAIN_PATH)

            def get_index(batch):
                return batch - 5278 // batch_size

        elif batch == 5534 // batch_size:
            X, mask, _, _ = get_test(TEST_PATH)

            def get_index(batch):
                return batch - 5534 // batch_size

        for batch_seq in range(batch_size):
            seq = batch * batch_size + batch_seq
            if not exists[seq]:
                idx = get_index(batch)
                for label in ssConvertString:
                    compute_tensor_jurtz(X[idx], mask[idx], batch, label, ini=batch_seq)
                break


if __name__ == "__main__":
    # main_saliencies()
    # main_SeqLogo()
    main_saliencies_jurtz()
    # repair_saliencies()


# DEPRECATED
def compute_tensor_saliency(X_am, X_pssm, args):
    if X_am.ndim == 2:
        X_am = X_am[None, ...]
        X_pssm = X_pssm[None, ...]

    model = load_model("modelQ8.h5")

    gradients = jacobian(model.outputs[0][:, :, ssConvertString.find(args.label)].flatten(),
                         wrt=[model.inputs[0], model.inputs[1]])
    get_gradients = K.function(inputs=[model.inputs[0], model.inputs[1], K.learning_phase()],
                               outputs=gradients)
    grads = get_gradients([X_am, X_pssm, 0])

    with open("saliencies/saliencies" + str(args.seq) + str(args.label) + ".pkl", 'wb') as f:
        pickle.dump(grads, f, protocol=2)


def main_saliencies():
    parser = argparse.ArgumentParser(description='Compute saliencies')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='class to which gradients are computed (default H)')
    parser.add_argument('--seq', type=int, default=0, metavar='seq',
                        help='sequence of which the gradient is calculated (default 0)')
    args = parser.parse_args()

    if args.seq is not None:
        first_seq = args.seq
        num_seqs = 1

        X_am, X_pssm, mask, labels = load_data("", first_seq=first_seq, num_seqs=num_seqs)
        compute_tensor_saliency(X_am, X_pssm, args)
