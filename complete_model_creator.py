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
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

warnings.filterwarnings('always')

#%% WANDB function creation
def wandb_block(params):
    config_directory = params
    trial_number = config_directory["name"]
    
    run = wandb.init(
        settings=wandb.Settings(x_stats_sampling_interval=params['sampling_interval'], 
                                x_disable_stats=False),
        # set the wandb project where this run will be logged
        name = f"Trial_{trial_number}",
        project = config_directory['project'],
        group = config_directory['project'],
        # track hyperparameters and run metadata with wandb.config
        config = config_directory
    )
    time.sleep(3.0)
    return run
      
#%% Neural Network Constructor
class NeuralNetworkConstructor:
    def __init__(self, inputs, total_categories=2, filters=64, conv_blocks=2, maxpooling_rate=2, bottleneck_length=2, 
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
                #rescalamos el residual
                residual=layers.Resizing(x_shape[1], 
                                         x_shape[2],
                                         interpolation="bilinear")(residual)
                
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

# MODEL BUILDER
class ModelBuilder:
    def __init__(self, trial, inputs, total_classes):
        # INITIAL CONFIGURATION
        self.project = 'Plant_Pathology_Classifier'
        self.group = 'Mini_Train'
        self.loss = 'categorical_crossentropy'
        self.metrics = 'val_loss'
        self.batch_size = 25
        self.sampling_interval = 20
        self.inputs=inputs
        self.total_categories = total_classes
        # PRIMARY OPTUNA SUGGESTIONS
        self.trial_number = trial
        self.conv_blocks = trial.suggest_int('conv_blocks', 1, 5)
        self.maxpooling_rate = trial.suggest_int('maxpooling_tate', 1, 5)
        self.total_length = self.maxpooling_rate * self.conv_blocks
        self.bottleneck_length = trial.suggest_int('bottleneck_length', 1, 5)
        self.filters = trial.suggest_int('filters', 16, 256, step=16)
        self.complete_length = self.conv_blocks*self.maxpooling_rate
        self.bottleneck_dropout = trial.suggest_float('bottleneck_dropout', 0.1, 0.5)
        self.optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "Nadam"])
        self.classifier_activation = 'softmax'
        
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
            activation_name = trial.suggest_categorical(f'BN_activation{i}', 
                                                        ['relu','tanh','leaky_relu','elu','silu','mish'])
            regularizer_name = trial.suggest_categorical(f'BN_regularizer{i}',
                                                        [None, 'L1L2', "L1", "L2"])
            self.bottleneck_activations[i] = activation_name
            self.bottleneck_regularizers[i] = regularizer_name
        
        for i in range(self.total_length):
            up_activation_name = trial.suggest_categorical(f'DW_activation{i}', 
                                                        ['relu','tanh','leaky_relu','elu','silu','mish'])
            up_regularizer_name = trial.suggest_categorical(f'DW_regularizer{i}',
                                                        [None, 'L1L2', "L1", "L2"])
            
            down_activation_name = trial.suggest_categorical(f'UW_activation{i}', 
                                                        ['relu','tanh','leaky_relu','elu','silu','mish'])
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
        outputs = Neural_network_constructor.complete_outputs()
        model = keras.Model(inputs=inputs,
                            outputs=outputs)
        model.compile(optimizer=self.optimizer,
                     loss=self.loss,
                     metrics=[self.metrics])
        return model
    
    def get_model(self):
        return self.model

# IMAGE GENERATOR
class ImageDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 image_dir: str,
                 image_id_column: str,
                 label_columns: List[str],
                 batch_size: int = 32,
                 target_size: Tuple[int, int] = (224, 224),
                 shuffle: bool = True,
                 preprocess_function: Optional[callable] = None):
        
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_id_column = image_id_column
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.preprocess_function = preprocess_function
        
        # Validate inputs
        self._validate_inputs()
        
        self.on_epoch_end()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory '{self.image_dir}' does not exist")
        
        if self.image_id_column not in self.dataframe.columns:
            raise ValueError(f"Image ID column '{self.image_id_column}' not found in dataframe")
        
        for col in self.label_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"Label column '{col}' not found in dataframe")
        
        if len(self.target_size) != 2:
            raise ValueError("target_size must be a tuple of (height, width)")
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        X = np.zeros((len(batch_df), *self.target_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_df), len(self.label_columns)), dtype=np.float32)
        
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            # Load and preprocess image
            img_path = os.path.join(self.image_dir, row[self.image_id_column])
            img_array = self.load_and_preprocess_image(img_path)
            X[i] = img_array
            
            # Get labels
            y[i] = row[self.label_columns].values
        
        return X, y
    
    def load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            # Load image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Apply custom preprocessing if provided
            if self.preprocess_function:
                img_array = self.preprocess_function(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return zero array as fallback
            return np.zeros((*self.target_size, 3), dtype=np.float32)
    
    def on_epoch_end(self) -> None:
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_class_distribution(self) -> pd.Series:
        """Get class distribution in the dataset."""
        return self.dataframe[self.label_columns].idxmax(axis=1).value_counts()
    
    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.dataframe)


def create_data_generators(csv_path: str,
                          image_dir: str,
                          image_id_column: str,
                          train_ratio: float = 0.6,
                          val_ratio: float = 0.3,
                          test_ratio: float = 0.1,
                          batch_size: int = 32,
                          target_size: Tuple[int, int] = (224, 224),
                          random_state: int = 42,
                          preprocess_function: Optional[callable] = None):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Identify label columns (all columns except image_id_column)
    label_columns = [col for col in df.columns if col != image_id_column]
    
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # First split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_ratio, 
        random_state=random_state, 
        stratify=df[label_columns].idxmax(axis=1)
    )
    
    # Second split: val vs test
    test_size_in_temp = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_size_in_temp, 
        random_state=random_state, 
        stratify=temp_df[label_columns].idxmax(axis=1)
    )
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Number of classes: {len(label_columns)}")
    
    # Create generators
    train_generator = ImageDataGenerator(
        dataframe=train_df,
        image_dir=image_dir,
        image_id_column=image_id_column,
        label_columns=label_columns,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=True,
        preprocess_function=preprocess_function
    )
    
    val_generator = ImageDataGenerator(
        dataframe=val_df,
        image_dir=image_dir,
        image_id_column=image_id_column,
        label_columns=label_columns,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False,
        preprocess_function=preprocess_function
    )
    
    test_generator = ImageDataGenerator(
        dataframe=test_df,
        image_dir=image_dir,
        image_id_column=image_id_column,
        label_columns=label_columns,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False,
        preprocess_function=preprocess_function
    )
    
    return train_generator, val_generator, test_generator, label_columns