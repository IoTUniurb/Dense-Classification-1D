# Imports
import keras

# Blocks
## Encoding
@keras.utils.register_keras_serializable()
class UNetEncoder1D(keras.Layer):
    def __init__(self, filters: int, **kwargs):
        # Parent constructor
        super().__init__(**kwargs)

        # Filters
        self.filters = filters

        # First Conv
        self.conv_1 = keras.layers.Conv1D(self.filters, 3, padding = 'same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()

        # Second Conv
        self.conv_2 = keras.layers.Conv1D(self.filters, 3, padding = 'same')
        self.bn_2 = keras.layers.BatchNormalization()
        self.relu_2 = keras.layers.ReLU()

    def call(self, inputs):
        # First Conv
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu_1(x)

        # Second Conv
        x = self.conv_2(inputs)
        x = self.bn_2(x)
        x = self.relu_2(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })

        return config

## Decoding
@keras.utils.register_keras_serializable()
class UNetDecoder1D(keras.Layer):
    def __init__(self, filters: int, **kwargs):
        # Parent constructor
        super().__init__(**kwargs)

        self.filters = filters

        # DeConv
        self.deconv = keras.layers.Conv1DTranspose(self.filters, 3, strides = 2, padding = 'same')
        self.relu = keras.layers.ReLU()

        # Concatenation
        self.concat = keras.layers.Concatenate(axis = -1)

        # First Conv
        self.conv_1 = keras.layers.Conv1D(self.filters, 3, padding = 'same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.ReLU()

        # Second Conv
        self.conv_2 = keras.layers.Conv1D(self.filters, 3, padding = 'same')
        self.bn_2 = keras.layers.BatchNormalization()
        self.relu_2 = keras.layers.ReLU()

    def call(self, inputs):
        x, skip = inputs

        # DeConv
        x = self.deconv(x)
        x = self.relu(x)

        # Concat
        x = self.concat([x, skip])

        # First Conv
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        # Second Conv
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
        })

        return config


# Network
def unet1d(input_shape: tuple, out_classes: int, filters: int = 8):
    inputs = keras.layers.Input(input_shape)

    e1 = UNetEncoder1D(filters=filters)(inputs)
    p1 = keras.layers.MaxPooling1D()(e1)

    ## Block 2
    e2 = UNetEncoder1D(filters=filters * 2)(p1)
    p2 = keras.layers.MaxPooling1D(pool_size = 2)(e2)

    ## Block 3
    e3 = UNetEncoder1D(filters=filters*4)(p2)
    p3 = keras.layers.MaxPooling1D(pool_size = 2)(e3)

    ## Block 4
    e4 = UNetEncoder1D(filters=filters*8)(p3)
    p4 = keras.layers.MaxPooling1D(pool_size = 2)(e4)

    ## Block 5
    e5 = UNetEncoder1D(filters=filters*16)(p4)



    ########## Decoder
    ## Block 1
    x = UNetDecoder1D(filters = filters * 8)([e5, e4])
    ## Block 2
    x = UNetDecoder1D(filters = filters * 4)([x, e3])
    ## Block 3
    x = UNetDecoder1D(filters = filters * 2)([x, e2])
    ## Block 4
    x = UNetDecoder1D(filters = filters)([x, e1])



    ########## Aggregator
    x = keras.layers.Conv1D(out_classes, 1, padding='same')(x)
    x = keras.layers.Softmax()(x)

    return keras.Model(inputs = inputs, outputs = x)
