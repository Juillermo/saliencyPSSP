import sys
import numpy as np
import glob

import utils

if len(sys.argv) < 2:
    sys.exit("Usage: python eval_avrg.py <predictions_path> [subset=test]")

predictions_path_all = glob.glob(sys.argv[1] + "*")

print("shape of metadata_path_all:", len(predictions_path_all))

import data

if len(sys.argv) == 3:
    subset = sys.argv[2]
    assert subset in ['train', 'valid', 'test', 'test_valid']
else:
    subset = 'test'

if subset == "test":
    _, mask, y, _ = data.get_test()
elif subset == "train":
    y = data.labels_train
    mask = data.mask_train
elif subset == "train_valid":
    y = data.labels
    mask = data.mask
else:
    y = data.labels_valid
    mask = data.mask_valid

acc_vec = np.zeros(len(predictions_path_all))
for i, predictions_path in enumerate(predictions_path_all):
    print(predictions_path)

    predictions = np.load(predictions_path)  # .ravel()
    acc = utils.proteins_acc(predictions, y, mask)

    print("Accuracy (%s) is: %.5f" % (subset, acc))
    acc_vec[i] = acc

print("Avg acc: " + str(np.mean(acc_vec)))
print("Std acc: " + str(np.std(acc_vec)))
