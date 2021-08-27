This repository contains the code for the paper:
>[Saliency Maps on CNNs for Protein Secondary Structure Prediction](https://ieeexplore.ieee.org/document/8683603)
 
The convolutional neural network (CNN) for protein secondary structure prediction (PSSP) has been built and trained using the framework [lasagne4bio](https://github.com/vanessajurtz/lasagne4bio) developed by [Vanessa Jurtz](https://github.com/vanessajurtz), and can be found in the  `secondary_proteins_prediction` folder of this repository.
My only additions to this folder are the CNN architectures `pureConv.py`, `HpureConv.py` and `pssmConv.py`, which are located at  `secondary_proteins_prediction/configurations`. These architectures were employed for the experiments in our paper and they correspond to the main architecture (`pureConv.py`) and variations predicting H values using aminoacids from one side (*left* or *right*) of a position only (`HpureConv.py`), or with only pssm input (`pssmConv.py`). The last two modified architectures also alter the training-prediction-evaluation pipeline, so I also had to make modifications in the files run for training, making predicitions, and evaluating models; resulting in the new files `Hpuretrain.py` and `pssmtrain.py` for training the *HpureConv* and *pssmConv* respectively, the new files `predict_side.py` and `pssmpredict.py` for making predictions on them, and `eval_avrg_side.py` for evaluating predictions on the *HpureConv* model.
Note that diagrams of the architecture of `pureConv.py` (in `jpeg` and `dia` formats) are also included in the `secondary_proteins_prediction/configurations` folder.

The code related to the creation of the saliency maps is located at the folder *test_saliency*. Some preliminary exploration of the code can be found at the jupyter notebook `Exploring Saliencies.ipynb` on the root directory. A short description of each file is included next:
- *saliency.py*: functions to produce the saliency maps (this is done in batches)
- *saliency_aggregation.py*: functions to aggregate the saliency maps along different dimensions
- *reparer.py*: scans saliency maps already generated or repairs the ones that had problems
- *results.py*: generates figures, statistics, and processed data
- *results_paper.py*: similar to *results.py*, but specific to the figures included in the paper
- *utils.py*: various basic utility functions and data


The protein dataset can be downloaded from [here](https://www.princeton.edu/%7Ejzthree/datasets/ICML2014/).

### Training models and evaluating them

In the *secondary_proteins_prediction/* folder:

1. Train model
    > python train.py pureConv

    (or alternatively `Hpuretrain.py HpureConv` / `pssmtrain.py pssmConv`)

2. After which use debug_metadata.py to find the best training epochs, found by validation performance
    > python debug_metadata.py <]topX> <metadata_path>

3. Use predict.py to gather predictions
    > python predict.py <metadata_path> [subset=test]

    (or `predict_side.py` / `pssmpredict.py`)

4. Use eval_avrg.py to evaluate the model and combining several predictions.
    > python eval_avrg.py <predictions_path> [subset=test]

    (or `eval_avrg_side.py` for *HpureConv* models)

### Computing saliencies from a trained model

1. Compute saliencies via `saliency.py`*.

2. `Exploring saliencies.ipnb` is a good way to get familiarised with other functions. Some of the functions found there may need data files containing different aggregations of the saliencies. These can be produced via `saliency_aggregation.py`.

3. Figures from the paper can be reproduced with the functions from `results_paper.py`, while other results not included in the paper can be found in `results.py`.

*Note that Computing saliencies requires a big amount of computational power, so this process may take very long times.

---------------------

## Installation (from the README of the *lasagne4bio* framework)

Please refer to [lasagne's](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04) for installation and setup of GPU environment on an Ubuntu 14.04 machine.

### Dependencies

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
