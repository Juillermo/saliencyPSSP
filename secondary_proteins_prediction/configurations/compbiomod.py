import lasagne
import numpy as np
import BatchNormLayer

batch_norm = BatchNormLayer.batch_norm
np.random.seed(1)

start_saving_at = 0
save_every = 1

epochs = 400
batch_size = 64

N_CONV_A = 50
F_CONV_A = 11

n_inputs = 42
num_classes = 8
seq_len = 700

optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.0025,
    2: 0.005,
    300: 0.0015,
    310: 0.001,
    320: 0.0005,
    330: 0.00025,
    340: 0.0001,
}


def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))

    # 2. Conv layer
    l_dim_a = lasagne.layers.DimshuffleLayer(l_in, (0, 2, 1))
    l_conv_a = lasagne.layers.Conv1DLayer(
        incoming=l_dim_a, num_filters=N_CONV_A, pad='same',
        filter_size=F_CONV_A, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_conv_a_b = batch_norm(l_conv_a)

    l_dim_b = lasagne.layers.DimshuffleLayer(
        l_conv_a_b, (0, 2, 1))
    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_dim_b, (batch_size * seq_len, N_CONV_A))

    # 3. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape_b, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out
