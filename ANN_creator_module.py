#%% Import libraries
# Warnings configuration
import warnings
warnings.filterwarnings('ignore')

# General Libraries
import tensorflow as tf
import numpy as np

# Neural Network Components
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K

warnings.filterwarnings('always')

#%% WANDB function creation
def wandb_block(params, sampling_interval, project, group):
    config_directory = params
    trial_number = config_directory["name"]
    run = wandb.init(
        settings=wandb.Settings(x_disable_stats=False, x_stats_sampling_interval = sampling_interval),
        # set the wandb project where this run will be logged
        name = f"Trial_{trial_number}",
        project = project,
        group = group,
        # track hyperparameters and run metadata with wandb.config
        config = config_directory
    )
    time.sleep(3.0)
    return run
      
#%% Neural Network Constructor
class NeuralNetworkConstructor:
    def __init__(self, inputs, filters=64, total_categories=2, conv_blocks=2, maxpooling_rate=2, bottleneck_length=2, 
                 downwards_activations=[None]*4, downwards_dropouts=[0.2,0.2], downwards_regularizers=[None]*4,
                 bottleneck_activations=[None]*2, bottleneck_dropout=0.2, bottleneck_regularizers=[None]*2,
                 upwards_activations=[None]*4, upwards_dropouts=[0.2,0.2], upwards_regularizers=[None]*4,
                 classifier_activation="relu"):
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
            selected_regularizer = "None"
        elif regularizer == "L1L2":
            selected_regularizer = f"keras.regularizers.{regularizer}(l1 = 1e-5, l2 = 1e-5)"
        else:
            selected_regularizer = f"keras.regularizers.{regularizer}(1e-5)"
        return selected_regularizer
        
    
    def downwards_path_iterator(self, block_inputs):
        x = block_inputs
        count=0
        for i in range(self.complete_length):
            current_filter = self.filters*(i//self.maxpooling_rate+1)
            current_activation = self.downwards_activations[i]
            current_regularizer = self.selected_regularizer(
                self.downwards_regularizers[i]
            )
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=eval(current_regularizer),
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
    
    def bottleneck_path_iterator(self, block_inputs):
        current_filter = self.filters*self.conv_blocks*2
        x = block_inputs
        for i in range(self.bottleneck_length):
            current_activation = self.bottleneck_activations[i]
            current_regularizer = self.selected_regularizer(
                self.bottleneck_regularizers[i]
            ) 
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=eval(current_regularizer),
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
            current_regularizer = self.selected_regularizer(
                self.upwards_regularizers[i]
            )
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
                h_crop=(r_shape[1]-x_shape[1])//2 - 0
                w_crop=(r_shape[2]-x_shape[2])//2 - 0
                #removemos los datos del residual para que igualen la dimension
                residual=layers.Cropping2D((h_crop,w_crop))(residual)
                # unimos los residuals con las activaciones actuales
                x = layers.Concatenate(axis=-1)([x, residual])
                x=layers.Dropout(self.downwards_dropouts[count-1],
                                 name=f'UpDO{count}')(x)
                current_filter= current_filter//2
                
            x = layers.Conv2D(filters=current_filter,
                              kernel_size=(3,3),
                              padding='valid',
                              activation=current_activation,
                              kernel_regularizer=eval(current_regularizer),
                              name=f'UpConv{i+1}')(x)
            
            x = layers.BatchNormalization(name=f'UpBN{i+1}')(x)
        upwards_output = x
        return upwards_output

    def complete_outputs(self):
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