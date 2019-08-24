# imports
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from time import localtime, strftime
from keras.applications.xception import preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend

# initialization
time = localtime()
img_width, img_height = 256, 256
checkpoint_filepath = strftime('Checkpoints/CNN_weights_%Y%m%d_%H%M%S.hdf5', time)
checkpoint_filepath_fine_tuned = strftime('Checkpoints/CNN_Fine_Tuned_weights_%Y%m%d_%H%M%S.hdf5', time)
log_dir = strftime('Logs/CNN_%Y%m%d_%H%M%S', time)
log_dir_fine_tuned = strftime('Logs/CNN_Fine_Tuned_%Y%m%d_%H%M%S', time)
features_test_filepath = strftime('CNN_features_test_%Y%m%d_%H%M%S', time)
prediction_filepath = strftime('../Predictions/pred_%Y%m%d_%H%M%S.csv', time)
train_data_dir = '../train'
test_data_dir = '../test'
epochs = 20
epochs_fine_tuned = 20
batch_size = 16

# augment data
train_validation_datagen = ImageDataGenerator(
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest',
	validation_split=0.2,
	preprocessing_function=preprocess_input)
train_generator = train_validation_datagen.flow_from_directory(
	train_data_dir,
	batch_size=batch_size,
	target_size=(img_width, img_height),
	class_mode='categorical',
	subset='training')
validation_generator = train_validation_datagen.flow_from_directory(
	train_data_dir,
	batch_size=batch_size,
	target_size=(img_width, img_height),
	class_mode='categorical',
	subset='validation')

# image format
if backend.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# callbacks
checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=log_dir)

# fit
model.fit_generator(
	train_generator,
	steps_per_epoch=train_generator.samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=validation_generator.samples // batch_size,
	callbacks=[checkpointer, tensorboard])

# fine tune
# model
model.load_weights(checkpoint_filepath)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# callbacks
checkpointer = ModelCheckpoint(checkpoint_filepath_fine_tuned, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=log_dir_fine_tuned)

# fit
model.fit_generator(
	train_generator,
	steps_per_epoch=train_generator.samples // batch_size,
	epochs=epochs_fine_tuned,
	validation_data=validation_generator,
	validation_steps=validation_generator.samples // batch_size,
	callbacks=[checkpointer, tensorboard])

# predictions
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	batch_size=1,
	target_size=(img_width, img_height),
	class_mode=None,
	shuffle=False)

features_test = model.predict_generator(test_generator, steps=test_generator.samples, verbose=1)

#print(features_test.shape)
#np.save(open(features_test_filepath, 'wb'), features_test)

# create csv file
label_strings = ['bobcat', 'chihuahua', 'collie', 'dalmatian', 'german_shepherd', 'leopard', 'lion', 'persian_cat', 'siamese_cat', 'tiger', 'wolf']
with open(prediction_filepath, 'w') as outfile:
	csvwriter = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	row_to_write = ['Id'] + [label for label in label_strings]
	csvwriter.writerow(row_to_write)
	for idx, prediction in enumerate(features_test):
		assert len(prediction) == len(label_strings)
		csvwriter.writerow([str(idx + 1)] + ["%.18f" % p for p in prediction])
