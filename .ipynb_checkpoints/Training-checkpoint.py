# Warnings configuration
import warnings
warnings.filterwarnings('ignore')

# General Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time

# Neural Network Components
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K

#Data preprocessing
from sklearn.model_selection import train_test_split
from PIL import Image

# Experiments Managers
import optuna
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

#Import the modules
from complete_model_creator import NeuralNetworkConstructor
from complete_model_creator import ModelBuilder
from complete_model_creator import  ImageDataGenerator

# Define directory path and parameters
data_dir = '/tf/keras_neural_network/Mis_Tests/plant-pathology-2020-fgvc7/images' # e.g., 'data/train'
image_size = (256//2, 384//2)
batch_size = 10

# Create a dataset for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/train',
    labels='inferred',  # Inferred from subdirectory names
    label_mode='categorical', # or 'binary' or 'int'
    image_size=image_size,
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.7,
    subset='training',
    seed=42 # for reproducibility
)

# Optional: Create a validation set by specifying validation_split and subset
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/test',
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.7,
    subset='training',
    seed=42,
)

# Preprocessing 
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Prepare for performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

number_of_classes = 4

def objective(trial):
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor ='val_loss',
                                                   min_delta = 0.001,
                                                   patience = 50, 
                                                   restore_best_weights = True,
                                                   mode = "auto",
                                                   verbose = 1,
                                                   baseline = None)

    builder = ModelBuilder(trial, 
                           inputs=keras.Input((image_size[0],
                                               image_size[1],
                                               3)),
                           total_classes=number_of_classes)
    
    model = builder.get_model()
    params = builder.get_params(1)
    
    run = wandb.init(
        project="Plant_Pathology_Test",
        group="mini_train",
        name=f"Trial_{trial.number}",
        config=params
    )
    
    # Train the model with data generators
    history = model.fit(
        train_ds,
        epochs=10, 
        batch_size=batch_size,  # Use the tuned batch_size
        verbose=0,
        validation_data=test_ds,
        callbacks=[
            WandbMetricsLogger(log_freq=1),
            early_stopping
        ]
    )
    
    # Finish wandb run
    run.finish()
    # Evaluate on validation set (not test set for hyperparameter optimization)
    val_loss, val_accuracy = model.evaluate(test_ds, verbose=0)
    return val_loss  # Optimize on validation loss

study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials = 30)