This repository contains the code for the paper:
>[Saliency Maps on CNNs for Protein Secondary Structure Prediction](https://ieeexplore.ieee.org/document/8683603)
 
The convolutional neural network (CNN) for protein secondary structure prediction (PSSP) has been built and trained using the framework [lasagne4bio](https://github.com/vanessajurtz/lasagne4bio) developed by [Vanessa Jurtz](https://github.com/vanessajurtz), and can be found in the *secondary_proteins_prediction* folder of this repository. My only additions to this folder regarding the training of CNNs are the CNN architectures *pureConv.py*, *HpureConv.py* and *pssmConv.py*, which are located at *secondary_proteins_prediction/configurations*. These architecures were employed for the experiments in the paper. The last two are modified architectures that can be trained and evaluated with the files *Hpuretrain.py*, *pssmtrain.py* and *Heval.py*, *pssmpredict.py*, respectively, and they correspond to architectures predicting H values from one side of a position only, or with only pssm input, respectively.

The code related to the creation of the saliency maps is located at the folder *test_saliency*. Some preliminary exploration of the code can be found at the jupyter notebook *"Exploring Saliencies.ipynb"* on the root directory.

The protein dataset can be downloaded from [here](https://www.princeton.edu/%7Ejzthree/datasets/ICML2014/).

---------------------

A relevant fragment from the README information of the *lasagne4bio* framework:

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
