import numpy as np
import matplotlib.pyplot as plt
import string
import importlib
#import lasagne as nn

from utils import toSeqLogo

#topX = 5
metadata_path = "../secondary_proteins_prediction/metadata/dump_pureConv-20180804-010835-47.pkl"
#metadata_path = "metadata/dump_stackedcomp-20180514-163759-42.pkl"
#metadata_path = "metadata/dump_compbiomod-20180512-201015-43.pkl"

print "Loading metadata file %s" % metadata_path
metadata = np.load(metadata_path)
print metadata['config_name']
#config = importlib.import_module("configurations.%s" % metadata['config_name'])

params = np.array(metadata['param_values'])

print "Params shape:", params.shape
for i, layer_params in enumerate(params):
    print "--Layer {:2d}: size {:4d}, shape {:s}, ".format(i, layer_params.size, layer_params.shape)

toSeqLogo(params[10][1].T)

if __name__ == "__main__":
    pass