# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Effnet.

Architecture: https://arxiv.org/abs/1801.06434
"""

import os
import tensorflow as tf
import numpy as np
from keras.models import Model,
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, SeparableConv2D, concatenate, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import matplotlib.pyplot as plt
import h5py


def ddb_b(X_input, growth, repeat=6, **kwargs):
    X = X_input
    for i in range(repeat):
        X = Conv2D(growth, (1,1), strides = 1, **kwargs)(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = SeparableConv2D(growth, (3,3), strides=1, **kwargs)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        
        X_input = concatenate([X_input, X], axis = -1)
    return X_input


def tinyDSOD(input_shape, l2):
    regul = regularizers.l2(l2)
    kwargs = {'padding':'same', 'kernel_regularizer':regul,'kernel_initializer':'glorot_uniform'}
    
    ### STEM ###
    # Convolution 1
    inp = Input(input_shape)
    X = Conv2D(64, (3,3), strides=2, **kwargs)(inp)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Convolution 2
    X = Conv2D(64, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Depth-wise seperable convolution 1
    X = SeparableConv2D(64, (3,3), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Convolution 3
    X = Conv2D(128, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Depth-wise seperable convolution 2
    X = SeparableConv2D(128, (3,3), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Pooling
    X = MaxPooling2D((2,2), strides=2, padding='same')(X)
    
    ### Extractor ###
    # Dense stage 0
    X = ddb_b(X, 32, repeat=4, **kwargs)
    
    # Transition layer 0
    X = Conv2D(128, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=2, padding='same')(X)
    
    # Dense stage 1
    X = ddb_b(X, 48, repeat = 6, **kwargs)
    
    # Transition layer 1
    X = Conv2D(128, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=2, padding='same')(X)
    
    # Dense stage 2
    X = ddb_b(X, 64, repeat=6, **kwargs)
    
    # Transition layer 2
    X = Conv2D(256, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Dense stage 3
    X = ddb_b(X, 80, repeat=6, **kwargs)
    
    # Transition layer 3
    X = Conv2D(64, (1,1), strides=1, **kwargs)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    
    ### Classification layers ###
    X = GlobalAveragePooling2D()(X)
    X = Dense(10, activation='softmax', kernel_regularizer=regul, kernel_initializer='glorot_uniform')(X)
    
    ### Create Model ###
    
    model = Model(inputs=inp, outputs=X, name='TinyDSOD_bb')
    
    return model


def training_scope(**kwargs):
    """Defines MobilenetV2 training scope.

    Usage:
         with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

      with slim.

      Args:
        **kwargs: Passed to mobilenet.training_scope. The following parameters
        are supported:
          weight_decay- The weight decay to use for regularizing the model.
          stddev-  Standard deviation for initialization, if negative uses xavier.
          dropout_keep_prob- dropout keep probability
          bn_decay- decay for the batch norm moving averages.

      Returns:
        An `arg_scope` to use for the mobilenet v2 model.
      """
    return lib.training_scope(**kwargs)


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):
  """Defines Mobilenet training scope.

  Usage:
     with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

     # the network created will be trainble with dropout/batch norm
     # initialized appropriately.
  Args:
    is_training: if set to False this will ensure that all customizations are
      set to non-training mode. This might be helpful for code that is reused
      across both training/evaluation, but most of the time training_scope with
      value False is not needed. If this is set to None, the parameters is not
      added to the batch_norm arg_scope.

    weight_decay: The weight decay to use for regularizing the model.
    stddev: Standard deviation for initialization, if negative uses xavier.
    dropout_keep_prob: dropout keep probability (not set if equals to None).
    bn_decay: decay for the batch norm moving averages (not set if equals to
      None).

  Returns:
    An argument scope to use via arg_scope.
  """
  # Note: do not introduce parameters that would change the inference
  # model here (for example whether to use bias), modify conv_def instead.
  batch_norm_params = {
      'decay': bn_decay,
      'is_training': is_training
  }
  if stddev < 0:
    weight_intitializer = slim.initializers.xavier_initializer()
  else:
    weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
      weights_initializer=weight_intitializer,
      normalizer_fn=slim.batch_norm), \
      slim.arg_scope([mobilenet_base, mobilenet], is_training=is_training),\
      safe_arg_scope([slim.batch_norm], **batch_norm_params), \
      safe_arg_scope([slim.dropout], is_training=is_training,
                     keep_prob=dropout_keep_prob), \
      slim.arg_scope([slim.conv2d], \
                     weights_regularizer=slim.l2_regularizer(weight_decay)), \
      slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
    return s


__all__ = ['training_scope', 'effnet']

