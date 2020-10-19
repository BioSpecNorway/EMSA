from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv1D, AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


def get_spectra_vgg(n_classes, n_wavenumbers, n_features=1, compiled=True):
    filter_size = 3
    filters_cnt = 12
    pooling = AveragePooling1D

    inp = Input(shape=(n_wavenumbers, n_features))
    conv1 = Conv1D(filters_cnt, filter_size)(inp)
    lerelu1 = LeakyReLU()(conv1)
    bn1 = BatchNormalization()(lerelu1)
    conv2 = Conv1D(filters_cnt, filter_size)(bn1)
    lerelu2 = LeakyReLU()(conv2)
    bn2 = BatchNormalization()(lerelu2)
    pool1 = AveragePooling1D(pool_size=8, strides=4)(bn2)

    pool_n = pool1
    filter_mult = 2
    n_blocks = 3
    for i in range(n_blocks):
        conv_n = Conv1D(filters_cnt * filter_mult, filter_size)(pool_n)
        lerelu_n = LeakyReLU()(conv_n)
        bn_n = BatchNormalization()(lerelu_n)
        conv_n2 = Conv1D(filters_cnt * filter_mult, filter_size)(bn_n)
        lerelu_n2 = LeakyReLU()(conv_n2)
        bn_n2 = BatchNormalization()(lerelu_n2)
        pool_n = pooling(pool_size=2, strides=2)(bn_n2)
        filter_mult *= 2

    output = Flatten()(pool_n)
    dropout = Dropout(0.5)(output)
    out = Dense(n_classes, activation='softmax')(dropout)

    model = Model(inputs=inp, outputs=out)
    if compiled:
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])

    return model