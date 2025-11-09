################
# Warnings configuration
import warnings
warnings.filterwarnings('ignore')

#General Libraries
import numpy as np
import pandas as pd

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import gc
from tensorflow.keras import backend as K


# Experiments Managers
import optuna
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
################
# Define directory path and parameters
data_dir = '/workspace/images'
factor=2
h = 256
w = 384
image_size = (h, w,3)
batch_size = 15

# Create a dataset for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/train',
    labels='inferred',  # Inferred from subdirectory names
    label_mode='categorical', # or 'binary' or 'int'
    image_size=image_size[:2],
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=True,
    seed=42 # for reproducibility
)

# Optional: Create a validation set by specifying validation_split and subset
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/test',
    labels='inferred',
    label_mode='categorical',
    image_size=image_size[:2],
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)
#########################
class_names = train_ds.class_names
num_of_classes = len(class_names)
###########################
def data_augmentation(x):
    x = layers.RandomColorJitter(
        [0,255],
        0.2,
        0.2,
        [0.4,5],
    )(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomRotation(0.1)(x)
    return x
############################
class PretrainedModelBuilder:
    def __init__(self, trial, num_of_classes=4, image_size=(h, w,3)):
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        self.inputs = keras.Input(image_size)
        self.base_model = DenseNet121(
            include_top=False,weights='imagenet',
            input_shape=image_size,
            input_tensor=self.inputs,pooling=None
        )
        self.base_model.trainable = False
        self.dense_layers = trial.suggest_int(
            'total_dense_layers',1,3)
        self.learning_rate = trial.suggest_float(
            'learning_rate',0.002,0.004
        )
        self.units = [None] * self.dense_layers
        self.dropouts = [None] * self.dense_layers
        self.activations = [None] * self.dense_layers
        self.regularizers = [None] * self.dense_layers
        self.act_categories = ['relu','leaky_relu','elu']
        self.opt_categories = ["Adam", "SGD", "RMSprop", "Nadam"]
        self.reg_categories = [None, 'L1', 'L2']
        self.optimizer = trial.suggest_categorical('optimizer', 
                                                   self.opt_categories)
        
        for i in range(self.dense_layers):
            self.units[i] = trial.suggest_int(
                f'activations_{i}',16,512,step=16)
            self.dropouts[i] = trial.suggest_float(
                f'dropouts_{i}',0.1,0.3)
            self.activations[i] = trial.suggest_categorical(
                f'activation_{i}', self.act_categories
            )
            self.regularizers[i] = trial.suggest_categorical(
                f'regularizer_{i}', self.reg_categories
            )
    def selected_regularizer(self, regularizer):
        if regularizer is None:
            return None
        elif regularizer == "L1L2":
            return regularizers.L1L2(l1=1e-5, l2=1e-5)
        elif regularizer == "L1":
            return regularizers.L1(1e-5)
        elif regularizer == "L2":
            return regularizers.L2(1e-5)
        
    def final_model(self):    
        x = self.inputs
        x = data_augmentation(x)
        x = layers.Rescaling(1./255)(x)
        x = self.base_model(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.25)(x) 
        
        for i in range(self.dense_layers):
            x = layers.Dense(
                units=self.units[i],
                activation=self.activations[i], 
                kernel_regularizer=self.selected_regularizer(
                    self.regularizers[i]
                ))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(rate=self.dropouts[i])(x)
        
        x = layers.Dense(
            units=self.num_of_classes, 
            activation='softmax', 
            name='clasification_layer')(x)
        outputs = x
        return outputs
        
    def _model_build(self):
        model = keras.Model(inputs=self.inputs, 
                            outputs=self.final_model())
        model.compile(
            loss='categorical_crossentropy',
            optimizer=eval(f"keras.optimizers.{self.optimizer}(learning_rate={self.learning_rate})"),
            metrics=['accuracy']
        )
        return model
    def get_params(self):
        params = {'dense_layers':self.dense_layers,
                  'dense_units':self.units,
                  'dense_activations':self.activations,
                  'regularizers':self.regularizers,
                  'dense_dropouts':self.dropouts,
                  'learning_rate':self.learning_rate,
                  'optimizer': self.optimizer}
        return params
#########################################
def objective(trial):
    # Initialize best tracking if not exists
    if not hasattr(objective, "best_val_loss"):
        objective.best_val_loss = float('inf')
        objective.best_trial_number = None

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0.0001,
                                                   patience=50, 
                                                   restore_best_weights=True,
                                                   mode="auto",
                                                   verbose=1,
                                                   baseline=None)

    builder = PretrainedModelBuilder(trial)
    params = builder.get_params()
    model = builder._model_build()

    # Start WandB run
    run = wandb.init(
        project='Plant_Diseases_DenseNet121',
        group='Full_Train_2',
        name=f'Trial_{trial.number}',
        config=params | {'best': 'no', 'trial_number': trial.number},
        reinit=True
    )
    
    try:
        history = model.fit(
            train_ds,
            epochs=1000, 
            batch_size=batch_size,
            verbose=0,
            validation_data=test_ds,
            callbacks=[
                WandbMetricsLogger(log_freq=20),
                early_stopping
            ]
        )
        
        val_loss, val_accuracy = model.evaluate(test_ds, verbose=0)
        
        # Determine if this is the best model
        is_best = val_loss < objective.best_val_loss
        if is_best:
            objective.best_val_loss = val_loss
            objective.best_trial_number = trial.number
        
        # Update WandB config with best status and final metrics
        wandb.config.update({
            'best': 'yes' if is_best else 'no',
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy,
            'best_val_loss_so_far': objective.best_val_loss,
            'best_trial_number_so_far': objective.best_trial_number
        }, allow_val_change=True)
        
        # Log final metrics
        wandb.log({
            'final_val_loss': val_loss,
            'final_val_accuracy': val_accuracy,
            'is_best_trial': is_best
        })
        
        final_val_loss = val_loss
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        final_val_loss = float('inf')
    
    finally:
        # Cleanup regardless of success or failure
        run.finish()
        
        # Clear TensorFlow session and free memory
        cleanup_gpu_memory(model)
        
        return final_val_loss

def cleanup_gpu_memory(model=None):
    """Clean up GPU memory and clear TensorFlow session"""
    try:
        # Delete model if provided
        if model is not None:
            del model
        
        # Clear TensorFlow session
        K.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Clear TensorFlow GPU memory
        if tf.config.list_physical_devices('GPU'):
            # For TensorFlow 2.x
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.reset_memory_stats('GPU:0')
                except RuntimeError as e:
                    print(f"GPU memory reset error: {e}")
        
        # Additional garbage collection
        gc.collect()
        
        print("GPU memory cleaned successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Initialize the best tracking variables
objective.best_val_loss = float('inf')
objective.best_trial_number = None

# Configure your study
study = optuna.create_study(direction="minimize")

# Add callback for additional cleanup between trials (optional)
def callback(study, trial):
    print(f"Trial {trial.number} completed. Best value: {study.best_value}")
    # Additional cleanup if needed
    cleanup_gpu_memory()

# Run the optimization
study.optimize(objective, n_trials=50, callbacks=[callback])