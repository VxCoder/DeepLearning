from keras.applications import VGG16
from keras import backend as K

model = VGG16(
    weights='imagenet',
    include_top=False

)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])
grads = K.gradients(loss, model.input)[0]
