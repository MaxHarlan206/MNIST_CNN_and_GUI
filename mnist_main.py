#import keras for data manipulation
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#import matplotlib for visualiztion of data
from matplotlib import pyplot

# DEFINE DATASET
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 0. GET AN UNDERSTANDING OF THE DATA, VISUALLY AND NUMERICALLY
# prints out a sample of our initial images
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i],cmap = pyplot.get_cmap('gray'))
    fig = pyplot.gcf()
    fig.canvas.set_window_title('Raw MNIST data')
pyplot.show()

# Show the number and dimensions of our train and test datasets
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# # Show the min, max, mean and average of the images.
# print(x_train.min(), x_train.max(), x_train.mean(), x_train.std())
# print(x_test.min(), x_test.max(), x_test.mean(), x_test.std())

# 1. LOAD DATA
def load_datasets():
    # Reshaping the data to have a single color channel for greyscale values
    global x_train
    global x_test
    global y_train
    global y_test
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # one hot encode target values
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test
load_datasets()

# 2. PREPARE PIXEL DATA
# def prepare_pixels(train, test):
#     # rescales the values of every pixel into the range 0-1
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
#     # prepare an iterators to scale images
#     global train_iterator
#     global test_iterator
#     train_iterator = datagen.flow(x_train, y_train, batch_size=32)
#     test_iterator = datagen.flow(x_test, y_test, batch_size=32)
#
#     print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
#     # confirm the scaling works
#     batchX, batchy = train_iterator.next()
#     print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# prepare_pixels(x_train, x_test)

# define cnn model
# def define_model():
#     global model
#     model = Sequential()
#     # Feature extraction
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     # Classification
#     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dense(10, activation='softmax'))
#     # compile model
#     opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
# define_model()
# # compile model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # fit model with generator
# model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=10)
# # save the model
# model.save('model.e10')
# print("model has successfully saved")
# # evaluate model
# _, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
# print('Test Accuracy: %.3f' % (acc * 100))

