#%% Import libraries

# General Libraries
import tensorflow as tf
import numpy as np
import time
import warnings

# Neural Network Components
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras import backend as K

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

warnings.filterwarnings('always')
      
#%% Neural Network Constructor
class NeuralNetworkConstructor:
    def __init__(self, inputs, total_categories=2, filters=64, conv_blocks=2, maxpooling_rate=2, bottleneck_length=2, 
                 downwards_activations=[None]*4, downwards_dropouts=[0.2,0.2], downwards_regularizers=[None]*4,
                 bottleneck_activations=[None]*2, bottleneck_dropout=0.2, bottleneck_regularizers=[None]*2,
                 upwards_activations=[None]*4, upwards_dropouts=[0.2,0.2], upwards_regularizers=[None]*4,
                 classifier_activation="softmax"):
        self.inputs = inputs
        self.conv_blocks = conv_blocks
        self.maxpooling_rate = maxpooling_rate
        self.bottleneck_length = bottleneck_length
        self.filters =  filters
        self.complete_length = conv_blocks*maxpooling_rate
        self.residuals = [None]*conv_blocks
        self.total_categories = total_categories
        self.downwards_activations = downwards_activations
        self.downwards_dropouts = downwards_dropouts 
        self.downwards_regularizers = downwards_regularizers 
        
        self.bottleneck_activations = bottleneck_activations 
        self.bottleneck_dropout = bottleneck_dropout 
        self.bottleneck_regularizers = bottleneck_regularizers 
        
        self.upwards_activations = upwards_activations 
        self.upwards_dropouts = upwards_dropouts  
        self.upwards_regularizers = upwards_regularizers
        self.classifier_activation = classifier_activation
    
    def selected_regularizer(self, regularizer):
        if regularizer is None:
            return None
        elif regularizer == "L1L2":
            return regularizers.L1L2(l1=1e-5, l2=1e-5)
        elif regularizer == "L1":
            return regularizers.L1(1e-5)
        elif regularizer == "L2":
            return regularizers.L2(1e-5)
        return None
        
    
    def downwards_path_iterator(self, block_inputs):
        x = block_inputs
        count=0
        for i in range(self.complete_length):
            current_filter = self.filters*(i//self.maxpooling_rate+1)
            current_activation = self.downwards_activations[i]
            current_regularizer = self.downwards_regularizers[i]
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=self.selected_regularizer(
                                  current_regularizer),
                              name=f'DownConv{i+1}')(x)
            x = layers.BatchNormalization(name=f'DownBN{i+1}')(x)
            
            if (i+1)%self.maxpooling_rate==0:
                self.residuals[count]=x
                x=layers.MaxPooling2D((2,2), padding='same',
                                      name=f'MaxPool{count+1}')(x)
                
                x=layers.Dropout(self.downwards_dropouts[count],
                                 name=f'DownDO{count+1}')(x)
                count+=1
                
        downwards_output = x
        return downwards_output

    def downwards_path_iterator_no_residuals(self, block_inputs):
        x = block_inputs
        count=0
        for i in range(self.complete_length):
            current_filter = self.filters*(i//self.maxpooling_rate+1)
            current_activation = self.downwards_activations[i]
            current_regularizer = self.downwards_regularizers[i]
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=self.selected_regularizer(
                                  current_regularizer),
                              name=f'DownConv{i+1}')(x)
            x = layers.BatchNormalization(name=f'DownBN{i+1}')(x)
            
            if (i+1)%self.maxpooling_rate==0:
                x=layers.MaxPooling2D((2,2), padding='same',
                                      name=f'MaxPool{count+1}')(x)
                
                x=layers.Dropout(self.downwards_dropouts[count],
                                 name=f'DownDO{count+1}')(x)
                count+=1
        downwards_output = x
        return downwards_output
    
    def bottleneck_path_iterator(self, block_inputs):
        current_filter = self.filters*self.conv_blocks*2
        x = block_inputs
        for i in range(self.bottleneck_length):
            current_activation = self.bottleneck_activations[i]
            current_regularizer = self.bottleneck_regularizers[i]
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=self.selected_regularizer(
                                  current_regularizer),
                              name=f'BotConv{i+1}')(x)
            x = layers.BatchNormalization(name=f'BotBN{i+1}')(x)
        x = layers.Dropout(self.bottleneck_dropout,
                           name="BN_Dropout")(x)
        bottleneck_output = x
        return bottleneck_output

    def upwards_path_iterator(self, block_inputs):
        current_filter = self.filters*self.conv_blocks*2
        count = 0
        x = block_inputs
        for i in range(self.complete_length):
            current_activation = self.upwards_activations[i]
            current_regularizer = self.upwards_regularizers[i]
            
            if i%self.maxpooling_rate==0:
                count+=1
                residual = self.residuals[self.conv_blocks-count]
                x = layers.Conv2DTranspose(filters=current_filter,
                                           kernel_size=(2,2),
                                           strides=(2,2),
                                           padding='same',
                                           name=f'UpSamp{i//self.maxpooling_rate+1}')(x)
                r_shape=residual.shape
                x_shape=x.shape
                #rescalamos el residual
                residual=layers.Resizing(x_shape[1], 
                                         x_shape[2],
                                         interpolation="bilinear")(residual)
                
                # unimos los residuals con las activaciones actuales
                x = layers.Concatenate(axis=-1)([x, residual])
                x=layers.Dropout(self.upwards_dropouts[count-1],
                                 name=f'UpDO{count}')(x)
                current_filter= current_filter//2
                
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=self.selected_regularizer(
                                  current_regularizer),
                              name=f'UpConv{i+1}')(x)
            
            x = layers.BatchNormalization(name=f'UpBN{i+1}')(x)
            upwards_output = x
        return upwards_output

    def segmentation_outputs(self):
        x=self.inputs
        x=self.downwards_path_iterator(x)
        x=self.bottleneck_path_iterator(x)
        x=self.upwards_path_iterator(x)
        x=layers.Conv2D(filters=self.total_categories,
                        kernel_size=(1,1),
                        padding='valid',
                        activation=self.classifier_activation,
                        name='segmentation_map')(x)
        outputs=x
        return outputs
        
    def classification_outputs(self, dense_units, dense_activation, dense_dropout):
        x=self.inputs
        x=self.downwards_path_iterator_no_residuals(x)
        x=self.bottleneck_path_iterator(x)
        x=layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(dense_units, activation=dense_activation)(x)
        x = layers.Dropout(dense_dropout)(x)
        x = layers.Dense(self.total_categories, 
                         activation=self.classifier_activation,
                         name='classifier')(x)
        outputs = x
        return outputs

# MODEL BUILDER
class ModelBuilder:
    def __init__(self, trial, inputs, total_classes):
        # INITIAL CONFIGURATION
        # self.activation_list = ['relu','tanh','leaky_relu','elu','silu','mish']
        self.activation_list = ['relu','tanh']
        self.project = 'Plant_Pathology_Classifier'
        self.group = 'Mini_Train'
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.batch_size = 20
        self.sampling_interval = 20
        self.inputs=inputs
        self.total_categories = total_classes
        # PRIMARY OPTUNA SUGGESTIONS
        self.trial_number = trial
        self.conv_blocks = trial.suggest_int('conv_blocks', 2, 3)
        self.maxpooling_rate = trial.suggest_int('maxpooling_rate', 2, 3)
        self.total_length = self.maxpooling_rate * self.conv_blocks
        self.bottleneck_length = trial.suggest_int('bottleneck_length', 2, 3)
        self.filters = trial.suggest_int('filters', 16, 256, step=16)
        self.complete_length = self.conv_blocks*self.maxpooling_rate
        self.bottleneck_dropout = trial.suggest_float('bottleneck_dropout', 0.1, 0.5)
        
        self.optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "Nadam"])
        self.classifier_activation = 'softmax'
        self.dense_units = trial.suggest_int('Dense_units', 16, 256, step=16)
        self.dense_activation = trial.suggest_categorical('Dense_activation', self.activation_list)
        self.dense_dropout = trial.suggest_float('Dense_dropout', 0.1, 0.5)
        # INITIALIZING THE CODER LISTS
        self.downwards_activations = [None]*self.complete_length
        self.downwards_dropouts = [None]*self.conv_blocks
        self.downwards_regularizers = [None]*self.complete_length
        
        # INITIALIZING THE BOTTLENECK LISTS
        self.bottleneck_activations = [None]*self.bottleneck_length
        self.bottleneck_regularizers = [None]*self.bottleneck_length
        
        # INITIALIZING THE DECODER LISTS
        self.upwards_activations = [None]*self.complete_length
        self.upwards_dropouts = [None]*self.conv_blocks
        self.upwards_regularizers = [None]*self.complete_length
       
        # SECONDARY OPTUNA SUGGESTIONS
        for i in range(self.bottleneck_length):
            activation_name = trial.suggest_categorical(f'BN_activation{i}', self.activation_list)
            regularizer_name = trial.suggest_categorical(f'BN_regularizer{i}',
                                                        [None, 'L1L2', "L1", "L2"])
            self.bottleneck_activations[i] = activation_name
            self.bottleneck_regularizers[i] = regularizer_name
        
        for i in range(self.total_length):
            up_activation_name = trial.suggest_categorical(f'DW_activation{i}', self.activation_list)
            up_regularizer_name = trial.suggest_categorical(f'DW_regularizer{i}',
                                                        [None, 'L1L2', "L1", "L2"])
            
            down_activation_name = trial.suggest_categorical(f'UW_activation{i}', self.activation_list)
            down_regularizer_name = trial.suggest_categorical(f'UW_regularizer{i}',
                                                        [None, 'L1L2', "L1", "L2"])
            
            self.upwards_activations[i] = up_activation_name
            self.upwards_regularizers[i] = up_regularizer_name
            self.downwards_activations[i] = down_activation_name
            self.downwards_regularizers[i] = down_regularizer_name

        for j in range(self.conv_blocks):
            down_dropout_value = trial.suggest_float(f'DW_dropout{j}',0.1,0.5)
            up_dropout_value = trial.suggest_float(f'UW_dropout{j}',0.1,0.5)
            self.upwards_dropouts[j] = up_dropout_value
            self.downwards_dropouts[j] = down_dropout_value
        
        self.model = self._build_model()
    
    def get_params(self, params_type=1):
        wandb_params = {
            'name': self.trial_number,
            'project': self.project,
            'group': self.group,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'batch_size': self.batch_size,
            'sampling_interval': self.sampling_interval
        }
        ann_params = {
            'filters': self.filters,
            'conv_blocks': self.conv_blocks, 
            'maxpooling_rate': self.maxpooling_rate,
            'bottleneck_length': self.bottleneck_length,
            'downwards_activations': self.downwards_activations,
            'downwards_dropouts': self.downwards_dropouts,
            'downwards_regularizers': self.downwards_regularizers,
            'bottleneck_activations': self.bottleneck_activations,
            'bottleneck_dropout': self.bottleneck_dropout,
            'bottleneck_regularizers': self.bottleneck_regularizers,
            'upwards_activations':self.upwards_activations,
            'upwards_regularizers':self.upwards_regularizers,
            'upwards_dropouts':self.upwards_dropouts,
            'classifier_activation': self.classifier_activation
        }
        if params_type==0:
            params = wandb_params
        elif params_type==1: 
            params = ann_params
        elif params_type==2:
            params = wandb_params | ann_params
        else:
            warnings.warn("params_type must be 0 (for wandb parameters), 1 (for ann parameters) or 2 (for all parameters)", 
                          UserWarning)
            params = None
        return params
        
    def _build_model(self):
        inputs = self.inputs
        total_categories = self.total_categories
        params = self.get_params(1)
        
        Neural_network_constructor = NeuralNetworkConstructor(inputs, total_categories, **params)
        outputs = Neural_network_constructor.classification_outputs(self.dense_units, 
                                                                    self.dense_activation,
                                                                    self.dense_dropout)
        model = keras.Model(inputs=inputs,
                            outputs=outputs)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=[self.metric])
        return model
    
    def get_model(self):
        return self.model