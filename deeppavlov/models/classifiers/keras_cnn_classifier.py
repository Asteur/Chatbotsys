from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

from deeppavlov.models.classifiers.intents.metrics import *


def keras_cnn_classifier(build_fn_params=None):
    """
    Build compiled model of shallow-and-wide CNN for multiclass multilabel classification
    Args:
        build_fn_params: parameters to compile the model (defaults from intents_snips)

    Returns:
        Compiled Keras model
    """
    params = build_fn_params or {}

    text_size = params.get("text_size", 15)
    embedding_size = params.get("embedding_size", 100)

    kernel_sizes_cnn = params.get("kernel_sizes_cnn", [1, 2, 3])
    filters_cnn = params.get("filters_cnn", 256)
    coef_reg_cnn = params.get("coef_reg_cnn", 1e-4)

    dropout_rate = params.get("dropout_rate", 0.5)
    dense_size = params.get("dense_size", 100)
    coef_reg_den = params.get("coef_reg_den", 1e-4)

    n_classes = params.get("n_classes", 2)
    n_classes = n_classes if n_classes != 2 else 1  # only 1 output neuron for binary classification!
    is_multilabel = params.get("is_multilabel", False)
    activation_function = "sigmoid" if is_multilabel else "softmax"

    lr = params.get("lr", 0.01)
    lr_decay = params.get("lr_decay", 0.1)
    optimizer = params.get("optimizer", "Adam")
    loss = params.get("loss", "binary_crossentropy")
    metrics = params.get("metrics", ["accuracy", "binary_accuracy", fmeasure])

    inp = Input(shape=(text_size, embedding_size))
    outputs = []
    for i in range(len(kernel_sizes_cnn)):
        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes_cnn[i],
                          activation=None,
                          kernel_regularizer=l2(coef_reg_cnn),
                          padding="same")(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation("relu")(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)

    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation=None,
                   kernel_regularizer=l2(coef_reg_den))(output)
    output = BatchNormalization()(output)
    act_output = Activation(activation_function)(output)
    model = Model(inputs=inp, outputs=act_output)

    model.compile(optimizer=eval(optimizer)(lr=lr, decay=lr_decay),
                  loss=loss,
                  metrics=metrics)

    return model
