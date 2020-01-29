#import libraries from keras
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#BUILDING THE CNN
#initializing CNN
classifier = Sequential()

#step 1 - convolution
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))

#step 2 - pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 - flattening
classifier.add(Flatten())

#step 4 - full connection
classifier.add(Dense(units = 100, activation = 'relu'))
# classifier.add(Dense(units = 120, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
# classifier.add(Dense(units = 132, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#DATA AUGMENTATION
#part 2 - fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator (rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory ('Training_Data/', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Test_Data/', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

#TRAINING MODEL
classifier.fit_generator(training_set, steps_per_epoch = 100, epochs = 4, validation_data = test_set, validation_steps = 10)




