from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ELU

from keras.models import Sequential, load_model
import os

classes = 2
batch_size = 24
image_shape = [64, 64, 3]

model = Sequential()
model.add(BatchNormalization(input_shape=image_shape))
model.add(Conv2D(24, (3, 3), strides=(1, 1), padding='valid'))
model.add(ELU())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid'))
model.add(ELU())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
model.add(ELU())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid'))
model.add(ELU())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Dense(classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if os.path.exists('model.h5'):
    model = load_model('model.h5')

generator = ImageDataGenerator(height_shift_range=0.3, width_shift_range=0.3)

images = generator.flow_from_directory('../images', target_size=(64, 64), batch_size=batch_size)

model.fit_generator(images, epochs=1, steps_per_epoch=300)

model.save('model.h5')

