This repository contains my code for my dissertation for the MSc in Artificial Intelligence in the University of Southampton:
>Applying Saliency Map Analysis to CNNs on Protein Secondary Structure Prediction
 
The network has been built and trained using the framework developed by [Vanessa Jurtz's](https://github.com/vanessajurtz) [lasagne4bio repository](https://github.com/vanessajurtz/lasagne4bio), which can be found in the *secondary_proteins_prediction* folder. I have mainly added new models to it (*pureConv.py* and *pssmConv.py* in *secondary_proteins_prediction/configurations*).

The saliency maps are built and aggregated in the folder *test_saliency*.

---------------------

# lasagne4bio

This repository provides code examples to train neural networks for 3 biological sequence analysis problems:

- subcellular localization
- secondary structure
- peptide binding to MHCII molecules

Please find detailed instructions in the respective directories.

## Data sets

All data sets are either included in the repositroy or links are provided to download them.

## Jupyter notebooks

In the directory `subcellular_localization` there are four tutorials on how to train four different types of neural networks for protein subcellular localization prediction:

 - Feedforward neural network
 - Convolutional neural network
 - Convolutional LSTM neural network
 - Convolutional LSTM neural network with attention mechanism

The dataset used for this tutorial is a reduced version of the original one, only with proteins shorter than 400 amino acids. This is done to save computational time, as here the main focus is to show how the network is built.

There is an additional tutorial on how to load the trained models and a comparison of their performances.

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


## Citation

Please cite the following when using our code as template:
...to be added...

## Contributors

Vanessa Isabell Jurtz, DTU Bioinformatics<br/>
Alexander Rosenberg Johansen, DTU Compute<br/>
Morten Nielsen, DTU Bioinformatics<br/>
Jose Juan Almagro Armenteros, DTU Bioinformatics<br/>
Henrik Nielsen, DTU Bioinformatics<br/>
Casper Kaae Sønderby, University of Copenhagen<br/>
Ole Winther, DTU Compute<br/>
Søren Kaae Sønderby, University of Copenhagen<br/>
