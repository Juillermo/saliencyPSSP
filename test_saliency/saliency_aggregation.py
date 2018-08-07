import os
import pickle
import argparse

import numpy as np

from utils import Jurtz_Data, convertPredictQ8Result2HumanReadable

WINDOW = 9


def calculate_SeqLogo(args):
    origin = os.getcwd()
    os.chdir('/scratch/grm1g17/saliencies')

    dater = Jurtz_Data()

    total = np.zeros((2 * WINDOW + 1, 42))  # WINDOW-size, n aminoacids + pssm
    for seq in range(args.num_seqs):
        try:
            with open("saliencies" + str(seq) + args.label + ".pkl", "rb") as f:
                saliency = np.array(pickle.load(f))

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
            print(str(seq) + " Not found")

    os.chdir(origin)
    with open("SeqLogo" + str(args.num_seqs) + args.label + ".pkl", "wb") as f:
        pickle.dump(total, f, protocol=2)


def calculate_points(args):
    points = []
    for seq, saliency in enumerate(saliencies):
        preds = convertPredictQ8Result2HumanReadable(predictions[seq])
        end_seq = int(sum(mask[seq]))
        for pos in range(end_seq):
            if True:#preds[pos] == args.label:
                total = np.zeros((2 * WINDOW + 1, 42))  # window-size, n aminoacids
                # Pre-window
                if pos > WINDOW:
                    init = pos - WINDOW
                    total[:WINDOW] += np.multiply(saliency[pos, init:pos, :], X[seq, init:pos])
                elif pos != 0:
                    init = WINDOW - pos
                    total[init:WINDOW] += np.multiply(saliency[pos, 0:pos, :], X[seq, 0:pos])

                # Window
                total[WINDOW] += np.multiply(saliency[pos, pos, :], X[seq, pos])

                # Post-window
                if pos + WINDOW + 1 <= end_seq:
                    end = pos + WINDOW + 1
                    total[WINDOW + 1:] += np.multiply(saliency[pos, pos + 1:end, :], X[seq, pos + 1:end])
                elif pos != end_seq:
                    end = end_seq
                    total[WINDOW + 1:-(pos + WINDOW + 1 - end)] += np.multiply(saliency[pos, pos + 1:end, :],
                                                                               X[seq, pos + 1:end])

                points.append(np.sum(total[:, 21:], axis=0))

    points = np.array(points)
    points = points / np.max(abs(points), axis=1)[:, None]
    print(points.shape)


def main_SeqLogo():
    parser = argparse.ArgumentParser(description='Compute SeqLogo from saliencies')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='class to which gradients are computed (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of sequences aggregated for SeqLogo (default 2)')
    args = parser.parse_args()

    if args.num_seqs is not None:
        calculate_SeqLogo(args)


def main_points():
    parser = argparse.ArgumentParser(description='Compute points from saliencies for clustering')
    parser.add_argument('--label', type=str, default='H', metavar='label',
                        help='predicted class of which points are extracted (default H)')
    parser.add_argument('--num-seqs', type=int, default=2, metavar='num_seqs',
                        help='number of sequences from which points are extracted (default 2)')
    args = parser.parse_args()

    if args.num_seqs is not None:
        calculate_points(args)


if __name__ == "__main__":
    main_SeqLogo()


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
