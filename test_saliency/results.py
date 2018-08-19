import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils import WINDOW, ssConvertString, pssmString_jurtz, aaString_jurtz, Jurtz_Data

FIGURES_PATH = "../thesis/Figures/"
SEQLOGO_PATH = "SeqLogo_data/"
CLASS_NAMES = "HGIEBLTS"
CLASS_COLOURS = {'L': 'blue', 'S': 'slateblue', 'T': 'dodgerblue',
                 'E': 'red', 'B': 'lightsalmon',
                 'H': 'green', 'G': 'limegreen', 'I': 'lightgreen'}

first = [int("E6", 16) / 255.0, 0, 0]
second = [0, int("D9", 16) / 255.0, 0]
third = [0, 0, int("FF", 16) / 255.0]
fourth = [0, 0, 0]
SEQLOGO_COLOURS = {'D': first, 'E': first,
                   'N': second, 'Q': second, 'S': second, 'G': second, 'T': second, 'Y': second,
                   'R': third, 'K': third, 'H': third,
                   'X': fourth, 'I': fourth, 'A': fourth, 'V': fourth, 'L': fourth, 'F': fourth}


def toSeqLogo(total):
    for row in range(len(total)):
        print(" ".join("{:.8f}".format(el) for el in total[row, 21:]))


def plot_outliers():
    dater = Jurtz_Data()
    X, labels, mask = dater.get_all_data()
    split_value = dater.split_value
    lengths_train = np.sum(mask[:split_value], axis=1)
    lengths_test = np.sum(mask[split_value:], axis=1)

    predictions_path = "../secondary_proteins_prediction/predictions/predictions_train_valid_pureConv-20180804-010835-47.npy"
    predictions = np.load(predictions_path)
    # print "train_val", predictions.shape

    predictions_path = "../secondary_proteins_prediction/predictions/predictionstest_pureConv-20180804-010835-47.npy"
    predictions2 = np.load(predictions_path)
    # print "test", predictions2[:-126].shape

    predictions = np.concatenate((predictions, predictions2[:-126]))

    # print "total", predictions.shape

    def calculate_seq_accuracy(predictions, labels):
        num_seq = len(mask)
        seq_len = predictions.shape[1]

        tot_acc = 0
        seq_acc = np.zeros(num_seq)
        for seq in range(num_seq):
            for pos in range(seq_len):
                if mask[seq, pos]:
                    if labels[seq, pos] == np.argmax(predictions[seq, pos]):
                        seq_acc[seq] += 1
                        tot_acc += 1
                else:
                    break

            seq_acc[seq] /= np.sum(mask[seq])

        # print tot_acc / np.sum(mask)
        return seq_acc

    seq_acc = calculate_seq_accuracy(predictions, labels)
    seq_acc_train = seq_acc[:split_value]
    seq_acc_test = seq_acc[split_value:]

    # print len(lengths_train[lengths_train > 300])
    # print len(lengths_test[lengths_test > 300])

    colors = np.zeros((len(labels), 3))
    for seq in range(len(labels)):
        for label in labels[seq, :int(np.sum(mask[seq]))]:
            if label in [ssConvertString.find('E'), ssConvertString.find('B')]:
                colors[seq, 0] += 1
            elif label in [ssConvertString.find('H'), ssConvertString.find('G'), ssConvertString.find('I')]:
                colors[seq, 1] += 1
            elif label in [ssConvertString.find('L'), ssConvertString.find('S'), ssConvertString.find('T')]:
                colors[seq, 2] += 1
        colors[seq] = colors[seq] / np.sum(mask[seq])

    def print_length_vs_acc(lengths, seqs, ax, label, colors):
        seq_len = 700

        for seq in range(len(seqs)):
            ax.plot(lengths[seq], seqs[seq], marker="X", linewidth=0, label="sequences", color=colors[seq])
        ax.plot(np.mean(seqs) * np.ones(seq_len), label="mean", color='orange')
        ax.set(title='Per-sequence accuracy (' + label + '), mean: {:.2f}'.format(np.mean(seqs)),
               ylabel="accuracy", xlabel="sequence length", ylim=[0, 1])
        blue_line = mlines.Line2D([], [], color='orange', label='mean accuracy')
        ax.legend(handles=[blue_line])

    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    print_length_vs_acc(lengths_train[:], seq_acc_train[:], ax[0], 'train', colors[:split_value])
    print_length_vs_acc(lengths_test, seq_acc_test, ax[1], 'test', colors[split_value:])

    # print_length_vs_acc(lengths_train[lengths_train > 300], seq_acc_train[lengths_train > 300], ax[1][0], 'train')
    # print_length_vs_acc(lengths_test[lengths_test > 300], seq_acc_test[lengths_test > 300], ax[1][1], 'test')

    plt.tight_layout()
    plt.savefig('thesis/Figures/per_seq_acc.eps', format='eps')


def plot_aa_pssm():
    num_seqs = 6018
    with open("aa_pssm" + str(num_seqs) + ".pkl", "rb") as f:
        aa_pssm = pickle.load(f)
        print("Total shape", aa_pssm.shape)

    aa_tot = aa_pssm[:, 0]
    pssm_tot = aa_pssm[:, 1]
    superiority = np.sum(pssm_tot) / np.sum(aa_tot)
    print(superiority)

    for i in range(len(aa_tot)):
        if aa_tot[i] == 0 and pssm_tot[i] == 0:
            aa_tot[i] = 0.01
            pssm_tot[i] = 0.01

    # plt.plot(aa_tot == 0)
    # plt.plot(pssm_tot == 0)

    base_diff = (pssm_tot - aa_tot)  # / superiority

    # from matplotlib.ticker import PercentFormatter

    fig, (ax2, ax) = plt.subplots(1, 2, figsize=(12, 3))
    min_x, max_x = 1, 6
    max_y = 1.1
    xlim = [min_x, max_x]
    ylim = [0, max_y]
    bins = np.linspace(min_x, max_x, 2 * (max_x - min_x) + 1)

    diff1 = pssm_tot / aa_tot
    ax.hist(diff1[base_diff > 0], bins=bins, density=True)  # , log=True)
    ax.set(xlim=xlim, title="superior pssm saliency scores: {:.1e}".format(len(diff1[base_diff > 0])),
           ylabel="fraction of total",
           xlabel="times superior", ylim=ylim)
    ax.yaxis.set(ticks=np.linspace(0, 1, 6), ticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

    diff2 = aa_tot / pssm_tot
    ax2.hist(diff2[base_diff < 0], bins=bins, density=True)  # , log=True)
    ax2.set(xlim=xlim, title="superior amino-acid saliency scores: {:.1e}".format(len(diff2[base_diff < 0])),
            ylabel="fraction of total",
            xlabel="times superior", ylim=ylim)
    ax2.yaxis.set(ticks=np.linspace(0, 1, 6), ticklabels=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.tight_layout()
    # plt.savefig(FIGURES_PATH + "aa_pssm.eps", format="eps")
    plt.show()
    # plt.savefig("../thesis/Figures/aa_pssm.eps", format="eps")

    return base_diff, pssm_tot, aa_tot


def collect_saliencies(folder="processed/"):
    num_seqs = 1
    saliencies = []

    for seq in range(num_seqs):
        if folder == "processed/":
            fname = folder + "saliencies" + str(seq) + ".npy"
            saliencies = np.load(fname)
        else:
            for target_class in ssConvertString:
                with open(folder + "saliencies" + str(seq) + target_class + ".pkl", "rb") as f:
                    saliency = np.array(pickle.load(f))
                    saliencies.append(saliency)
                saliencies = np.array(saliencies)

    print("Saliency sequence shape", saliencies.shape)
    return saliencies


def plot_sliding_saliencies():
    saliencies = collect_saliencies()

    def plot_singles(saliencies):
        ini, end = 2, 8
        plot_len = end - ini
        fig, axes = plt.subplots(plot_len / 2, 2, figsize=(15, plot_len))
        for pos in range(ini, end):
            ax1 = axes[(pos - ini) % 3][(pos - ini) // 3]
            tot_plot = saliencies[pos, 5]

            ax1.plot(tot_plot[:, :])
            ax1.set(title="saliency map of position " + str(pos))
            ax1.xaxis.set(ticks=range(19), ticklabels=range(-9, 10))

        plt.tight_layout()
        # plt.savefig(FIGURES_PATH+"sliding.eps", format="eps")
        plt.show()

    def plot_for_clustering(saliencies):
        end_seq = len(saliencies)
        ini, end = 2, 8
        plot_len = end - ini
        fig, axes = plt.subplots(plot_len, 2, figsize=(15, 2 * plot_len))
        for pos in range(ini, end):
            (ax1, ax2) = (axes[pos - ini][0], axes[pos - ini][1])
            tot_plot = saliencies[pos, 5]

            ax1.plot(tot_plot[:, 21:])
            ax1.xaxis.set(ticks=range(19), ticklabels=range(-9, 10))

            ax2.plot(np.sum(tot_plot[:, 21:], axis=0))
            ax2.xaxis.set(ticks=range(21), ticklabels=pssmString_jurtz)
        plt.show()

    plot_singles(saliencies)


def plot_single_sequence():
    saliencies = collect_saliencies()
    preds = "EEEEEEEEELGHHHGLGTGLLSEEE"
    labels = "SEEEEEEEELHHHHHSGGGGSSLEE"
    aas = "IXAGVEVQLTDDFFADEKSISENYV"

    final_pos = 119
    ini_pos = 95
    end = final_pos + WINDOW + 1
    ini = ini_pos - WINDOW

    tot_sal = np.zeros((8, end - ini, 42))
    for pos in range(ini_pos, final_pos):
        # print(saliencies[pos].shape)
        # beg = ini_pos - ini
        # fin = beg + 2 * WINDOW + 1
        # print(beg, fin, tot_sal[:, beg:fin, :].shape, tot_sal.shape)

        tot_sal[:, (pos - ini_pos):(pos - ini_pos + 2 * WINDOW + 1)] += saliencies[pos]

    def plot_heatmap():
        fig, ax = plt.subplots(figsize=(7, 6.65))
        vmax = np.max(abs(tot_sal[5, :, 21:]))
        cax = ax.imshow(tot_sal[5, WINDOW:-WINDOW, 21:].T, cmap='PiYG', vmin=-vmax, vmax=vmax)
        # fig.colorbar(cax)

        ax.yaxis.set(ticks=range(len(pssmString_jurtz)), ticklabels=pssmString_jurtz)
        ax.xaxis.set(ticks=range(final_pos - ini_pos + 1),
                     ticklabels=[preds[i] + "\n" + el + "\n" + str(ini_pos + i) for i, el in
                                 enumerate(labels)])
        ax.margins(0)
        ax2 = ax.twiny()
        ax2.xaxis.set(ticks=range(final_pos - ini_pos + 1), ticklabels=aas)

        plt.tight_layout()
        # fig.savefig(FIGURES_PATH + "sample_Hclass.eps", format='eps')
        plt.show()

        toSeqLogo(tot_sal[5])

    # plot_heatmap()

    tot_sal = np.sum(tot_sal, axis=2)

    def plot_lines():
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for j, label in enumerate(CLASS_NAMES):
            ax.plot(tot_sal[ssConvertString.find(label), WINDOW:-WINDOW].T, marker='.', label=label,
                    color=CLASS_COLOURS[label])

        ax.legend(loc="upper left")

        ax.xaxis.set(ticks=range(final_pos - ini_pos + 1),
                     ticklabels=[preds[i] + "\n" + el + "\n" + str(ini_pos + i) for i, el in
                                 enumerate(labels)])
        colors = [CLASS_COLOURS[el] for el in preds]
        for color, tick in zip(colors, ax.xaxis.get_major_ticks()):
            tick.label1.set_color(color)  # set the color property

        ax.margins(0)
        ax2 = ax.twiny()
        ax2.xaxis.set(ticks=range(final_pos - ini_pos + 1), ticklabels=aas)
        # colors = [SEQLOGO_COLOURS[el] for el in aas]
        # for color, tick in zip(colors, ax2.xaxis.get_major_ticks()):
        # tick.label2.set_color(color)  # set the color property

        plt.tight_layout()
        fig.savefig(FIGURES_PATH + "sample_8classes.eps", format='eps')
        plt.show()

    plot_lines()


def plot_sheer_sequences():
    totals = np.zeros((8, 19, 42))
    for i, target_class in enumerate(ssConvertString):
        num_seqs = 6018
        with open("SeqLogo_data/SeqLogo" + str(num_seqs) + target_class + ".pkl", "rb") as f:
            totals[i] = pickle.load(f)

    totals += totals[:, ::-1, :]
    totals /= 2

    def plot_sheer_class(totals):
        fig, axes = plt.subplots(1, 3, figsize=(6 * 3 / 2, 6.65 / 2))
        for i, target_class in enumerate(["H", "E", "L"]):
            ax = axes[i]
            tot_sal = totals[ssConvertString.find(target_class)]
            vmax = np.max(abs(tot_sal[..., 21:]))
            cax = ax.imshow(tot_sal[..., 21:].T, cmap='PiYG', vmin=-vmax, vmax=vmax)
            # fig.colorbar(cax)

            ax.set(title="Class " + target_class)
            ax.yaxis.set(ticks=range(len(pssmString_jurtz)), ticklabels=pssmString_jurtz)
            ax.xaxis.set(ticks=range(2 * WINDOW + 1),
                         ticklabels=range(-WINDOW, WINDOW + 1))
            ax.margins(0)

        plt.tight_layout()
        fig.savefig(FIGURES_PATH + "class_agg_class.eps", format='eps')
        plt.show()

    def plot_sheer_aa(totals):
        for i, aa in enumerate(pssmString_jurtz):
            if aa == "G":
                fig, axes = plt.subplots(1, 2, figsize=(12, 3))
                ax = axes[0]
                ax1 = axes[1]
                for j, label in enumerate(CLASS_NAMES):
                    ax.plot(totals[ssConvertString.find(label), :, aaString_jurtz.find(aa)], label=label, marker='.',
                            color=CLASS_COLOURS[label])
                    ax1.plot(totals[ssConvertString.find(label), :, pssmString_jurtz.find(aa) + 21], label=label,
                             marker='.',
                             color=CLASS_COLOURS[label])

                ax.legend(loc='right')
                # vmax = np.max(abs(totals[:, :, :21]))
                # ax.set(title=aa)
                # ax.set(ylim=[-vmax, vmax], title="Aggregated saliencies (one-hot): " + aa)
                ax.xaxis.set(ticks=range(19), ticklabels=range(-WINDOW, WINDOW + 1))
                # ax.yaxis.set(ticks=[])
                ax.margins(0)

                # vmax = np.max(abs(totals[:, :, 21:]))
                ax1.legend(loc='right')
                # ax1.set(ylim=[-vmax, vmax])  # , title="Aggregated saliencies (pssm): " + aa)
                ax1.xaxis.set(ticks=range(19), ticklabels=range(-WINDOW, WINDOW + 1))
                # ax1.yaxis.set(ticks=[])
                ax1.margins(0)

                plt.tight_layout()
                fig.savefig(FIGURES_PATH + "class_agg_aa.eps")
                plt.show()

    def plot_sheer_class_aa(totals):
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 3))
        ax2 = axes2[0]
        ax21 = axes2[1]

        legend_names = [el for el in CLASS_NAMES]
        omg_cosa = []
        for i, target_class in enumerate(legend_names):
            tot_aa = np.sum(abs(totals[ssConvertString.find(target_class), :, :21]), axis=1)
            tot_pssm = np.sum(abs(totals[ssConvertString.find(target_class), :, 21:]), axis=1)

            ax3 = ax2.twinx()
            ax3.plot(tot_aa, marker='.', label=target_class, color=CLASS_COLOURS[target_class])
            ax3.yaxis.set(ticks=[])

            ax22 = ax21.twinx()
            ax22.plot(tot_pssm, marker='.', label=target_class, color=CLASS_COLOURS[target_class])
            ax22.yaxis.set(ticks=[])

            omg_cosa.append(mlines.Line2D([], [], color=CLASS_COLOURS[target_class], marker='.', label=target_class))

        ax2.xaxis.set(ticks=range(19), ticklabels=range(-WINDOW, WINDOW + 1))
        ax2.margins(0)
        ax2.legend(omg_cosa, legend_names)
        ax2.yaxis.set(ticks=[])
        # ax2.legend()

        ax21.xaxis.set(ticks=range(19), ticklabels=range(-WINDOW, WINDOW + 1))
        ax21.margins(0)
        ax21.legend(omg_cosa, legend_names)  # , legend_names)
        ax21.yaxis.set(ticks=[])

        plt.tight_layout()
        fig2.show()
        fig2.savefig(FIGURES_PATH + "sheer_class_aa.eps")

    # plot_sheer_class(totals)
    plot_sheer_aa(totals)
    # plot_sheer_class_aa(totals)
    return totals


plot_outliers()
# b, p, a = plot_aa_pssm()
# sal = collect_saliencies()
# plot_sliding_saliencies()
# plot_single_sequence()
# totals = plot_sheer_sequences()

if __name__ == "__main__":
    # main()
    ""
