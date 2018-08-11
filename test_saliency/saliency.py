import pickle
import argparse
import importlib

import numpy as np
import theano
import theano.tensor as T
import keras.backend as K
from keras.models import load_model
import lasagne as nn

from utils import ssConvertString, Jurtz_Data

BATCH_SIZE = 64
PATH_SALIENCIES = "/scratch/grm1g17/saliencies/"

def compute_single_saliency(X_batch, seq_len, batch_seq, label, sym_x, inference):
    gradients = theano.gradient.jacobian(inference[batch_seq, :seq_len, ssConvertString.find(label)],
                                         wrt=sym_x)
    get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

    grads = get_gradients(X_batch)
    return np.array(grads)

def compute_broken_saliency(X_batch, seq_len, batch_seq, label, sym_x, inference):
    # FIRST HALF
    gradients = theano.gradient.jacobian(inference[batch_seq, :seq_len // 2, ssConvertString.find(label)],
                                         wrt=sym_x)
    get_gradients = theano.function(inputs=[sym_x], outputs=gradients)
    grads = np.array(get_gradients(X_batch))
    del get_gradients
    del gradients

    # SECOND HALF
    gradients = theano.gradient.jacobian(inference[batch_seq, seq_len // 2:seq_len, ssConvertString.find(label)],
                                         wrt=sym_x)
    get_gradients = theano.function(inputs=[sym_x], outputs=gradients)
    grads2 = np.array(get_gradients(X_batch))

    # TODO: FIX THE OVERLAPPING PART AT THE JOINT POINT
    return np.concatenate((grads[:, batch_seq, seq_len], grads2[:, batch_seq, seq_len]), axis=0)


def compute_tensor_jurtz(X_batch, mask_batch, batch, label, ini=0):
    """
    Computes the saliencies of a batch for a certain label, starting at batch index ini.

    Inputs:
        X_batch: batch of length 64, with its corresponding mask batch
        batch: batch number (absolute, for labeling)
        label: one of the 8 classes (see ssConvertString)
        ini: which sequence of the batch to start from

    Outputs:
        Saves individual saliencies in separated pickle files, properly labelled"""

    metadata_path = "dump_pureConv-20180804-010835-47.pkl"
    metadata = np.load(metadata_path)
    config_name = metadata['config_name']
    config = importlib.import_module("%s" % config_name)
    print("Using configurations: '%s'" % config_name)
    l_in, l_out = config.build_model()

    sym_x = T.tensor3()
    inference = nn.layers.get_output(
        l_out, sym_x, deterministic=True)
    nn.layers.set_all_param_values(l_out, metadata['param_values'])

    print("Computing saliencies")
    for batch_seq in range(ini, BATCH_SIZE):

        seq_len = int(np.sum(mask_batch[batch_seq]))
        fname = "saliencies{:4d}{:s}.pkl".format(BATCH_SIZE * batch + batch_seq, label)

        try:
            grads = compute_single_saliency(X_batch=X_batch, seq_len=seq_len, batch_seq=batch_seq, label=label, sym_x=sym_x, inference=inference)

        except Exception as err: # IF GPU OUT OF MEMORY
            print(err)
            grads = compute_broken_saliency(X_batch=X_batch, seq_len=seq_len, batch_seq=batch_seq, label=label, sym_x=sym_x, inference=inference)

        with open(PATH_SALIENCIES + fname, 'wb') as f:
            pickle.dump(grads, f, protocol=2)
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
from data_managing import load_data

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
