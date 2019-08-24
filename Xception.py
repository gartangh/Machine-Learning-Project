# imports
import csv
import numpy as np
from time import localtime, strftime
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# initialization
time = localtime()
img_width, img_height = 299, 299
checkpoint_filepath = strftime('Checkpoints/Xception_weights_%Y%m%d_%H%M%S.hdf5', time)
checkpoint_filepath_fine_tuned = strftime('Checkpoints/Xception_Fine_Tuned_weights_%Y%m%d_%H%M%S.hdf5', time)
log_dir = strftime('Logs/Xception_%Y%m%d_%H%M%S', time)
log_dir_fine_tuned = strftime('Logs/Xception_Fine_Tuned_%Y%m%d_%H%M%S', time)
features_test_filepath = strftime('Xception_features_test_%Y%m%d_%H%M%S', time)
prediction_filepath = strftime('../Predictions/pred_%Y%m%d_%H%M%S.csv', time)
train_data_dir = '../train'
test_data_dir = '../test'
epochs = 10
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

# model
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.1)(x)
#x = Dense(128, activation='relu')(x)
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
	layer.trainable = False

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
for i, layer in enumerate(base_model.layers):
	print(i, layer.name)

for layer in model.layers[:126]:
	layer.trainable = False
for layer in model.layers[126:]:
	layer.trainable = True

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
