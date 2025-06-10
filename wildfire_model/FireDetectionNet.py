from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

import numpy as np

class FireDetectionNet:
    def __init__(self):
        self.model_dict=None

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        model.add(SeparableConv2D(16, (7, 7), padding="same",
                                  input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(SeparableConv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # second set of FC => RELU layers
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
    @staticmethod
    def save_model_dict(model, filename):
        input_shape= model.input_shape
        current_shape=input_shape

        model_dict = {}
        for i, layer in enumerate(model.layers):
            output_shape = layer.compute_output_shape(current_shape)
            current_shape = output_shape
            layer_info = {
                'layer_type': layer.__class__.__name__,
                'config': layer.get_config(),
                'output_shape': output_shape,
                'number_of_parameters': layer.count_params()
            }
            model_dict[f'layer_{i}'] = layer_info

        # Save the model_dict as a .npy file
        np.save(filename, model_dict)
        print(f"Model dictionary saved as {filename}")

        # Store as an attribute
        FireDetectionNet.model_dict = model_dict

    @staticmethod
    def load_model_dict(filename='model_dict.npy'):
        # Load the model_dict from a .npy file
        model_dict = np.load(filename, allow_pickle=True).item()
        print(f"Model dictionary loaded from {filename}")
        return model_dict

        # Store as an attribute
        FireDetectionNet.model_dict = model_dict