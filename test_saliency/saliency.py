import pickle
import time
import argparse
import importlib

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import keras.backend as K
from keras.models import load_model
import lasagne as nn

from utils import decode, ssConvertMap, ssConvertString, convertPredictQ8Result2HumanReadable, Jurtz_Data
from data_managing import load_data

BATCH_SIZE = 64


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

        with open("saliencies_jurtz/saliencies{:4d}{:s}.pkl".format(BATCH_SIZE * batch + ini, label),
                  'wb') as f:
            pickle.dump(tot_grads, f, protocol=2)

        ini += 1

    print("Compute saliencies")
    for batch_seq in range(ini, BATCH_SIZE):
        seq_len = int(np.sum(mask_batch[batch_seq]))
        gradients = theano.gradient.jacobian(inference[batch_seq, :seq_len, ssConvertString.find(label)],
                                             wrt=sym_x)
        get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

        grads = get_gradients(X_batch)
        grads = np.array(grads)

        with open("saliencies_jurtz/saliencies{:4d}{:s}.pkl".format(BATCH_SIZE * batch + batch_seq, label),
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

    if args.batch is not None:
        first_seq = args.batch * BATCH_SIZE
        dater = Jurtz_Data()

        X_batch, mask_batch = dater.get_batch_from_seq(first_seq)
        compute_tensor_jurtz(X_batch, mask_batch, args.batch, args.label)



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

    # IT ACTUALLY FAILS, NEEDS TO TAKE DROPOUTS INTO ACCOUNT, SEE PREVIOUS VERSION
    gradients = theano.gradient.jacobian(model.outputs[0][:, :, ssConvertString.find(args.label)].flatten(),
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
