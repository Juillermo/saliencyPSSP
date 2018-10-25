import sys
import numpy as np
import glob

import utils

if len(sys.argv) < 2:
    sys.exit("Usage: python eval_avrg.py <predictions_path> [subset=test]")

predictions_path_all = glob.glob(sys.argv[1] + "*")

print("shape of metadata_path_all")
print(len(predictions_path_all))

mybool = False
for predictions_path in predictions_path_all:
    print(predictions_path)
    if not mybool:
        predictions = np.load(predictions_path)  # .ravel()
        mybool = True
    else:
        predictions = predictions + np.load(predictions_path)  # .ravel()
print("shape of predictions")
print(predictions.shape)
print(predictions.max())
print("Beginning: ", predictions[0, :15])
print("End: ", predictions[0, -15:])

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

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if y[i][j] == 5:
            y[i][j] = 1
        else:
            y[i][j] = 0

acc = utils.proteins_acc(predictions, y, mask)

print("Accuracy (%s) is: %.5f" % (subset, acc))
