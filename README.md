This repository contains the code for the paper:
>[Saliency Maps on CNNs for Protein Secondary Structure Prediction](https://ieeexplore.ieee.org/document/8683603)
 
The convolutional neural network (CNN) for protein secondary structure prediction (PSSP) has been built and trained using the framework [lasagne4bio](https://github.com/vanessajurtz/lasagne4bio) developed by [Vanessa Jurtz](https://github.com/vanessajurtz), and can be found in the *secondary_proteins_prediction* folder of this repository. My only additions to this folder regarding the training of CNNs are the CNN architectures *pureConv.py*, *HpureConv.py* and *pssmConv.py*, which are located at *secondary_proteins_prediction/configurations*. These architecures were employed for the experiments in the paper. The last two are modified architectures that can be trained and evaluated with the files *Hpuretrain.py*, *pssmtrain.py* and *Heval.py*, *pssmpredict.py*, respectively, and they correspond to architectures predicting H values from one side of a position only, or with only pssm input, respectively.

The code related to the creation of the saliency maps is located at the folder *test_saliency*. Some preliminary exploration of the code can be found at the jupyter notebook *"Exploring Saliencies.ipynb"* on the root directory. A short description of each file is included next:
- *data_managing.py*: processes data to keep it in handier formats
- *pureConv.py*: file containing the CNN architecture employed in the paper
- *saliency.py*: functions to produce the saliency maps (this is done in batches)
- *saliency_aggregation.py*: functions to aggregate the saliency maps along different dimensions
- *reparer.py*: scans saliency maps already generated or repairs the ones that had problems
- *results.py*: generates figures, statistics, and processed data
- *results_paper.py*: same as *results.py*, but for the figures included in the paper
- *utils.py*: various basic utility functions and data


The protein dataset can be downloaded from [here](https://www.princeton.edu/%7Ejzthree/datasets/ICML2014/).

---------------------

# Relevant fragments from the README information of the *lasagne4bio* framework:
## Reproducing results

### Installation

Please refer to [lasagne's](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04) for installation and setup of GPU environment on an Ubuntu 14.04 machine.

### Training models

Train model
>> python train.py T31

After which use debug\_metadata.py to find the best training epochs, found by validation performance
>> debug_metadata.py

Use predict.py to gather predictions
>> predict.py

Use eval\_avrg.py to evaluate the model and combining several predictions.
>> eval_avrg.py

## Dependencies

All code was written in python programming language version 2.7. Neural networks are implemented using the lasagne library, please find installation instructions here: [https://lasagne.readthedocs.io/en/latest/user/installation.html](https://lasagne.readthedocs.io/en/latest/user/installation.html).<br/>

The libraries used in this code are:

- argparse
- cPickle
- csv
- datetime
- gc
- glob
- gzip
- importlib
- itertools
- lasagne
- math
- matplotlib
- numpy
- operator
- os
- platform
- scipy
- sklearn
- string
- subprocess
- sys
- theano
- time
