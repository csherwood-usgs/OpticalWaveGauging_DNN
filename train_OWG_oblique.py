


# import libraries
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import zipfile
import os
import json
import time, datetime
from glob import glob
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from utils import *

from keras.metrics import mean_absolute_error

# mean absolute error
def mae_metric(in_gt, in_pred):
    return mean_absolute_error(div*in_gt, div*in_pred)



#==============================================================
## script starts here
if __name__ == '__main__':

	# load the user configs
	with open(os.getcwd()+os.sep+'conf'+os.sep+'config.json') as f:
		config = json.load(f)

	print(config)
	# config variables
	imsize    = int(config["img_size"])
	num_epochs = int(config["num_epochs"]) ##100
	test_size = float(config["test_size"])
	batch_size = int(config["batch_size"])
	height_shift_range = float(config["height_shift_range"])
	width_shift_range = float(config["width_shift_range"])
	rotation_range = float(config["rotation_range"])
	samplewise_std_normalization = config["samplewise_std_normalization"]
	horizontal_flip = config["horizontal_flip"]
	vertical_flip = config["vertical_flip"]
	samplewise_center = config["samplewise_center"]
	shear_range = float(config["shear_range"])
	zoom_range = float(config["zoom_range"])
	steps_per_epoch = int(config["steps_per_epoch"])
	dropout_rate = float(config["dropout_rate"])
	epsilon = float(config["epsilon"])
	min_lr = float(config["min_lr"])
	factor = float(config["factor"])
	input_image_format = config["input_image_format"]
	input_csv_file = config["input_csv_file"]
	category = config["category"] ##'H'
	fill_mode = config["fill_mode"]
	input_image_format = config["input_image_format"]
	image_dir = config["image_dir"]
	sitename = config["sitename"]


	IMG_SIZE = (imsize, imsize) ##(128, 128)

	base_dir = os.path.normpath(os.getcwd()+os.sep+'train')

	counter=4

	df = pd.read_csv(os.path.join(base_dir, input_csv_file))

	if input_image_format == 'jpg':
		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}.jpg'.format(x)))
	elif input_image_format == 'png':
		df['path'] = df['id'].map(lambda x: os.path.join(base_dir,image_dir,'{}.png'.format(x)))

	df['exists'] = df['path'].map(os.path.exists)
	print(df['exists'].sum(), 'images found of', df.shape[0], 'total')
	df.columns = [k.strip() for k in df.columns]


	mean = df[category].mean()
	div = 2*df[category].std()
	df['zscore'] = df[category].map(lambda x: (x-mean)/div)

	df.dropna(inplace = True)


	df['category'] = pd.cut(df[category], 10)


	new_df = df.groupby(['category']).apply(lambda x: x.sample(500, replace = True)).reset_index(drop = True)

	print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])

	train_df, valid_df = train_test_split(new_df,
									   test_size = test_size, #0.33,
									   random_state = 2018,
									   stratify = new_df['category'])
	print('train', train_df.shape[0], 'validation', valid_df.shape[0])


	im_gen = ImageDataGenerator(samplewise_center=samplewise_center, ##True,
								  samplewise_std_normalization=samplewise_std_normalization, ##True,
								  horizontal_flip = horizontal_flip, ##False,
								  vertical_flip = vertical_flip, ##False,
								  height_shift_range = height_shift_range, ##0.1,
								  width_shift_range = width_shift_range, ##0.1,
								  rotation_range = rotation_range, ##10,
								  shear_range = shear_range, ##0.05,
								  fill_mode = fill_mode, ##'reflect', #'nearest',
								  zoom_range= zoom_range) ##0.2)

	train_gen = gen_from_df(im_gen, train_df,
								 path_col = 'path',
								y_col = 'zscore',
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = batch_size) ##64)

	valid_gen = gen_from_df(im_gen, valid_df,
								 path_col = 'path',
								y_col = 'zscore',
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = batch_size) ##64)

	test_X, test_Y = next(gen_from_df(im_gen,
								   valid_df,
								 path_col = 'path',
								y_col = 'zscore',
								target_size = IMG_SIZE,
								 color_mode = 'grayscale',
								batch_size = len(df))) ##1000


	t_x, t_y = next(train_gen)

	train_gen.batch_size = batch_size

	weights_path='out'+os.sep+'model'+str(counter)+os.sep+'batch'+str(batch_size)+os.sep+category+'_weights_model'+str(counter)+'_'+str(batch_size)+'batch_best_'+sitename+'.hdf5'

	model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1,
							 save_best_only=True, mode='min', save_weights_only = True)


	reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=10, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)
	earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=15)
	callbacks_list = [model_checkpoint, earlystop, reduceloss_plat]

	base_model = InceptionResNetV2(input_shape =  t_x.shape[1:], include_top = False, weights = None)

	print ("[INFO] Training optical wave gauge")
	print(t_x.shape[1:])
	OWG = Sequential()
	OWG.add(BatchNormalization(input_shape = t_x.shape[1:]))
	OWG.add(base_model)
	OWG.add(BatchNormalization())
	OWG.add(GlobalAveragePooling2D())
	OWG.add(Dropout(dropout_rate)) ##0.5
	OWG.add(Dense(1, activation = 'linear' ))

	OWG.compile(optimizer = 'adam', loss = 'mse',
						   metrics = [mae_metric])
	OWG.summary()
	history = OWG.fit_generator(train_gen, validation_data = (test_X, test_Y), epochs = num_epochs, steps_per_epoch= steps_per_epoch, callbacks = callbacks_list)
	OWG.load_weights(weights_path)


	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('Loss_vs_epochs.png', dpi=300, bbox_inches='tight')
	plt.close()


	print ("[INFO] Testing optical wave gauge")

	# the model predicts zscores - recover value using pop. mean and standard dev.
	pred_Y = div*OWG.predict(test_X, batch_size = train_gen.batch_size, verbose = True)+mean
	test_Y = div*test_Y+mean

	print ("[INFO] Creating plots ")

	fig, ax1 = plt.subplots(1,1, figsize = (6,6))
	ax1.plot(test_Y, pred_Y, 'k.', label = 'predictions')
	ax1.plot(test_Y, test_Y, 'r-', label = 'actual')
	ax1.legend()

	ax1.set_xlabel('Actual')
	ax1.set_ylabel('Predicted')

	plt.savefig('im'+str(IMG_SIZE[0])+'_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch_'+sitename+'.png', dpi=300, bbox_inches='tight')


	plt.close('all')

	rand_idx = np.random.choice(range(test_X.shape[0]), 9)
	fig, m_axs = plt.subplots(3, 3, figsize = (16, 32))
	for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):
	  c_ax.imshow(test_X[idx, :,:,0], cmap = 'gray')
	  c_ax.set_title('%0.3f\nPredicted: %0.3f' % (test_Y[idx], pred_Y[idx]))
	  c_ax.axis('off')


	fig.savefig('im'+str(IMG_SIZE[0])+'_waveperiod_predictions_model'+str(counter)+'_'+str(num_epochs)+'epoch'+str(train_gen.batch_size)+'batch_'+sitename+'.png', dpi=300, bbox_inches='tight')

	counter += 1
