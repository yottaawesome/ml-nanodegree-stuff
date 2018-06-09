from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()
'''
Formula: Number of Parameters in a Convolutional Layer

The number of parameters in a convolutional layer depends on the supplied values of filters, kernel_size, and input_shape. Let's define a few variables:

    K - the number of filters in the convolutional layer
    F - the height and width of the convolutional filters
    D_in - the depth of the previous layer

Notice that K = filters, and F = kernel_size. Likewise, D_in is the last value in the input_shape tuple.

Since there are F*F*D_in weights per filter, and the convolutional layer is composed of K filters, the total number of weights in the convolutional layer is K*F*F*D_in. Since there is one bias term per filter, the convolutional layer has K biases. Thus, the number of parameters in the convolutional layer is given by K*F*F*D_in + K.

Formula: Shape of a Convolutional Layer

The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

    K - the number of filters in the convolutional layer
    F - the height and width of the convolutional filters
    S - the stride of the convolution
    H_in - the height of the previous layer
    W_in - the width of the previous layer

Notice that K = filters, F = kernel_size, and S = stride. Likewise, H_in and W_in are the first and second value of the input_shape tuple, respectively.

The depth of the convolutional layer will always equal the number of filters K.

If padding = 'same', then the spatial dimensions of the convolutional layer are the following:

    height = ceil(float(H_in) / float(S))
    width = ceil(float(W_in) / float(S))

If padding = 'valid', then the spatial dimensions of the convolutional layer are the following:

    height = ceil(float(H_in - F + 1) / float(S))
    width = ceil(float(W_in - F + 1) / float(S))

Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 32)        896                      (none, height=100, width=100, depth=32)
=================================================================
Total params: 896
Trainable params: 896
Non-trainable params: 0
_________________________________________________________________
'''