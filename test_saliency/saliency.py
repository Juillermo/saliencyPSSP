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


def compute_complex_saliency(X_batch, mask_batch, batch_seq, inference, sym_x, batch, label):
    seq_len = int(np.sum(mask_batch[batch_seq]))
    try:
        sym_y = inference[batch_seq, :seq_len, ssConvertString.find(label)]
        grads = compute_single_saliency(X_batch=X_batch, sym_x=sym_x, sym_y=sym_y)
        grads = grads[:seq_len, batch_seq, :seq_len]

    except Exception as err:
        # IF GPU OUT OF MEMORY
        print(err)
        try:
            # FIRST HALF
            sym_y = inference[batch_seq, :seq_len // 2, ssConvertString.find(label)]
            grads1 = compute_single_saliency(X_batch=X_batch, sym_x=sym_x, sym_y=sym_y)
            grads1 = grads1[:seq_len // 2, batch_seq, :seq_len]
        except Exception:
            print(err)
            print("XXXXXXX Is it in the first part?")

        try:
            # SECOND HALF
            sym_y = inference[batch_seq, seq_len // 2:seq_len, ssConvertString.find(label)]
            grads2 = compute_single_saliency(X_batch=X_batch, sym_x=sym_x, sym_y=sym_y)
            grads2 = grads2[seq_len // 2:seq_len, batch_seq, :seq_len]
        except Exception:
            print(err)
            print("XXXXXXXX Or in the second?")

        grads = np.concatenate((grads1, grads2), axis=0)

    assert grads.shape[0] == grads.shape[
        1], "You see, there is a problem over here. {:d} is not equal to {:d} for sequence with length {:d}, and I don't know why this is happening. I did concatenate grad1 with shape {:s} with grads2 with shape {:s}, but the fact is that I got grads with shape {:s}. Does this make any sense?".format(
        grads.shape[0],
        grads.shape[1], seq_len, str(grads1.shape), str(grads2.shape), str(grads.shape))
    fname = "saliencies{:4d}{:s}.pkl".format(BATCH_SIZE * batch + batch_seq, label)
    with open(PATH_SALIENCIES + fname, 'wb') as f:
        pickle.dump(grads, f, protocol=2)


def compute_single_saliency(X_batch, sym_x, sym_y):
    gradients = theano.gradient.jacobian(sym_y, wrt=sym_x)
    get_gradients = theano.function(inputs=[sym_x], outputs=gradients)

    grads = get_gradients(X_batch)
    return np.array(grads)


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
        compute_complex_saliency(X_batch=X_batch, sym_x=sym_x, batch_seq=batch_seq, mask_batch=mask_batch,
                                 inference=inference, batch=batch, label=label)
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
    main_saliencies_jurtz()