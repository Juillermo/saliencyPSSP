# RNNProteins

(All of this *README.md* file is from the original project [lasagne4bio](https://github.com/vanessajurtz/lasagne4bio), except for additions of my own included in the **Training models** section)

World record on cb513 dataset using cullpdb+profile\_6133\_filtered (68.7% Q8 with best single performing model, 70.2% Q8 with model ensemble), available at:
http://www.princeton.edu/~jzthree/datasets/ICML2014/

By Alexander Rosenberg Johansen

previous best single model results: 68.3% Q8 by: [Deep CNF](http://www.nature.com/articles/srep18962)

## Reproducing results

### Installation

Please refer to [lasagne's](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04) for installation and setup of GPU environment on an Ubuntu 14.04 machine.

### Training models

*Note that diagrams of the architecture of `pureConv.py` (in `jpeg` and `dia` formats) can be found in the `secondary_proteins_prediction/configurations` folder.

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

## Best network elaborated

https://github.com/vanessajurtz/lasagne4bio/blob/master/secondary\_proteins\_prediction/configurations/T31.py

1. InputLayer
2. 3x ConvLayer(InputLayer, filter\_size=3-5-7) + Batch Normalization
4. DenseLayer1([ConcatLayer, InputLayer]) + Batch Normalization
5. LSTMLayerF(DenseLayer1, Forward)
6. LSTMLayerB([DenseLayer1, LSTMLayerF], Backward)
7. DenseLayer2([LSTMLayerF, LSTMLayerB], dropout=0.5)
8. OutLayer(DenseLayer2)

Gradients are further normalized if too large and probabilities cutted. RMSProps is used and L2=0.0001
