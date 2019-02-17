from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import numpy as np 
import math
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from Match import *

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.optimizers import Adam



class ARCII:

    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        n1d_kernel_count = 32
        n1d_kernel_size = 15
        num_conv2d_layers = 2
        n2d_kernel_counts = [32,32]
        h2d_kernel_size = [[15,15],[15,15]]
        n2d_mpool_size = [[2,2],[2,2]]
        dropout_rate = 0.1
        
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )
        conv1d_layer = Conv1D(n1d_kernel_count, n1d_kernel_size, padding='same')

        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        q_embed = emb_layer(seq1)
        d_embed = emb_layer(seq2)

        q_conv1 = conv1d_layer(q_embed)
        d_conv1 = conv1d_layer(d_embed)
        cross = Match(match_type='plus')([q_conv1, d_conv1])
        z = Reshape((max_sequence_length, max_sequence_length, -1))(cross)

        for i in range(num_conv2d_layers):
            z = Conv2D(filters=n2d_kernel_counts[i], kernel_size=h2d_kernel_size[i], padding='same', activation='relu')(z)
            z = MaxPooling2D(pool_size=(n2d_mpool_size[i][0], n2d_mpool_size[i][1]))(z)

        #dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])([conv2d, dpool_index])
        pool1_flat = Flatten()(z)
        pool1_flat_drop = Dropout(dropout_rate)(pool1_flat)
        dense = Dense(300,activation='relu')(pool1_flat_drop)
        dense = Dropout(0.4)(dense)
        dense = Dense(30,activation='relu')(dense)
        dense = Dropout(0.4)(dense)
        pred = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        ada = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['acc'])

        return model