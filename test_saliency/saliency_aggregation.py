import os
import pickle
import argparse

import numpy as np

from utils import Jurtz_Data, convertPredictQ8Result2HumanReadable

WINDOW = 9
SALIENCIES_SCRATCH_PATH = '/scratch/grm1g17/saliencies'


def calculate_SeqLogo(args):
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)

    dater = Jurtz_Data()

    fail_seqs = 0
    total = np.zeros((2 * WINDOW + 1, 42))  # WINDOW-size, n aminoacids + pssm
    for seq in range(args.num_seqs):
        try:
            try:
                fname = "saliencies" + str(seq) + args.label + ".pkl"
                with open(fname, "rb") as f:
                    saliency = np.array(pickle.load(f))
            except OSError:
                fname = "saliencies{:4d}{:s}.pkl".format(seq, args.label)
                with open(fname, "rb") as f:
                    saliency = np.array(pickle.load(f))

            if saliency.ndim != 3:
                os.remove(fname)
                print("File "+fname+" deleted")
                raise OSError("saliency badly formatted")

            X_seq, mask_seq = dater.get_sequence(seq)
            end_seq = int(sum(mask_seq))
            for pos in range(end_seq):
                # Pre-WINDOW
                if pos > WINDOW:
                    init = pos - WINDOW
                    total[:WINDOW] += np.multiply(saliency[pos, init:pos, :], X_seq[init:pos])
                elif pos != 0:
                    init = WINDOW - pos
                    total[init:WINDOW] += np.multiply(saliency[pos, 0:pos, :], X_seq[0:pos])

                # Window
                total[WINDOW] += np.multiply(saliency[pos, pos, :], X_seq[pos])

                # Post-WINDOW
                if pos + WINDOW + 1 <= end_seq:
                    end = pos + WINDOW + 1
                    total[WINDOW + 1:] += np.multiply(saliency[pos, pos + 1:end, :], X_seq[pos + 1:end])
                elif pos != end_seq:
                    end = end_seq
                    total[WINDOW + 1:-(pos + WINDOW + 1 - end)] += np.multiply(saliency[pos, pos + 1:end, :],
                                                                               X_seq[pos + 1:end])
            print(seq)
        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)
    print("SeqLogo of " + str(args.num_seqs - fail_seqs) + " elements")
    with open("SeqLogo_data/SeqLogo" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(total, f, protocol=2)


def calculate_points(args):
    origin = os.getcwd()
    os.chdir(SALIENCIES_SCRATCH_PATH)

    dater = Jurtz_Data()

    fail_seqs = 0
    points = []
    for seq in range(args.num_seqs):
        try:
            try:
                with open("saliencies" + str(seq) + args.label + ".pkl", "rb") as f:
                    saliency = np.array(pickle.load(f))
            except OSError:
                with open("saliencies{:4d}{:s}.pkl".format(seq, args.label), "rb") as f:
                    saliency = np.array(pickle.load(f))

            X_seq, mask_seq = dater.get_sequence(seq)
            end_seq = int(sum(mask_seq))
            for pos in range(end_seq):
                total = np.zeros((2 * WINDOW + 1, 42))  # window-size, n aminoacids
                # Pre-WINDOW
                if pos > WINDOW:
                    init = pos - WINDOW
                    total[:WINDOW] += np.multiply(saliency[pos, init:pos, :], X_seq[init:pos])
                elif pos != 0:
                    init = WINDOW - pos
                    total[init:WINDOW] += np.multiply(saliency[pos, 0:pos, :], X_seq[0:pos])

                # Window
                total[WINDOW] += np.multiply(saliency[pos, pos, :], X_seq[pos])

                # Post-WINDOW
                if pos + WINDOW + 1 <= end_seq:
                    end = pos + WINDOW + 1
                    total[WINDOW + 1:] += np.multiply(saliency[pos, pos + 1:end, :], X_seq[pos + 1:end])
                elif pos != end_seq:
                    end = end_seq
                    total[WINDOW + 1:-(pos + WINDOW + 1 - end)] += np.multiply(saliency[pos, pos + 1:end, :],
                                                                               X_seq[pos + 1:end])
                points.append(np.sum(total[:, 21:], axis=0))
            print(seq)
        except OSError:
            fail_seqs += 1
            print(str(seq) + " Not found")

    os.chdir(origin)

    points = np.array(points)
    print(points.shape)

    print("Clustering points of " + str(args.num_seqs - fail_seqs) + " elements")
    with open("points" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(points, f, protocol=2)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate saliencies either by sheer addition or compute points for clustering')

    parser.add_argument('--func', type=str, default='sheer', metavar='func',
                        help='Function: sheer addition of saliencies (default), or calculate points for clustering')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='class from which to analyse the saliencies (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of sequences aggregated for SeqLogo (default 2)')
    args = parser.parse_args()

    if args.func is 'sheer':
        calculate_SeqLogo(args)
    elif args.func is 'points':
        calculate_points(args)
    else:
        raise ValueError('Function "'+args.func+'" not recognized, try with "sheer" or "points"')


if __name__ == "__main__":
    main()


# DEPRECATED

def calculate_SeqLogo_fang(args):
    _, _, mask, _ = load_data("", num_seqs=args.num_seqs)
    del _
    WINDOW = 9

    total = np.zeros((2, 2 * WINDOW + 1, 21))  # one-hot/pssm, WINDOW-size, n aminoacids
    for seq in range(args.num_seqs):
        with open("saliencies/saliencies" + str(seq) + args.label + ".pkl", "rb") as f:
            saliencies = np.array(pickle.load(f))

        end_seq = int(sum(mask[seq]))
        for pos in range(end_seq):
            # Pre-WINDOW
            if pos > WINDOW:
                init = pos - WINDOW
                total[:, :WINDOW] += saliencies[:, pos, 0, init:pos, :]
            elif pos != 0:
                init = WINDOW - pos
                total[:, init:WINDOW] += saliencies[:, pos, 0, 0:pos, :]

            # Window
            total[:, WINDOW] = saliencies[:, pos, 0, pos, :]

            # Post-WINDOW
            if pos + WINDOW + 1 <= end_seq:
                end = pos + WINDOW + 1
                total[:, WINDOW + 1:] = saliencies[:, pos, 0, pos + 1:end, :]
            elif pos != end_seq:
                end = end_seq
                total[:, WINDOW + 1:-(pos + WINDOW + 1 - end)] = saliencies[:, pos, 0, pos + 1:end, :]

    with open("SeqLogo" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(total, f, protocol=2)


# DEPRECATED

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
