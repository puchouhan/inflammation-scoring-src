""" 
Copyright (c) 2018, Fabian Heinemann, Gerald Birk, Tanja Schönberger, Birgit Stierstorfer; Boehringer Ingelheim Boehringer Ingelheim Pharma GmbH & Co KG
All rights reserved.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials # provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


import os
import numpy as np
from skimage import data, io, filters, transform
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import pandas as pd
import shutil
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import load_model
from keras import backend as K
from keras import applications
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import warnings
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Set parameters for input and output of the training below
"""

# Base path
base_path = "./"

# Path of training and validation images (relative to base path)
train_path = "training/"
val_path = "val/"

# Path of DL model
model_path = base_path + "model/"
model_name = "Inflammation_Model.h5"

# Path of result data
results_base_path = model_path

# Image dimensions (299x299 for InceptionV3 based nets)
img_width, img_height = 299, 299

# Set this to true to split off validation data from val (otherwise exisiting data in <val_path> will be used)
do_val_split = False

# Fraction of data to be used as validation data
val_fraction = 0.15

# number of epochs to train
n_epochs = 45

# Batch size
batch_size = 32

def split_validation_data(image_classes_list, base_path, train_path, val_path):
	""" Will take a random part of <val_fraction> from data in 
		subfolders of <train_path> and move to a subfolder <val_path>
		
		Args:
			image_classes_list: list of strings matching labels and subfolder names with images
			base_path: Base path of model
			train_path: Base train path under <base_path>
			val_path: Base val path under <base_path>
		Returns:	
			None
	""" 
	
	# Make sure to move exiting validation data to train otherwise the split will not work	
	# Delete old folders in val_path        
	for image_class in image_classes_list:
		if os.path.exists(base_path + val_path + image_class):
			if len(next(os.walk(base_path + val_path + image_class))[2]) > 0:
				print("Please move data from val to train first.")                
				break
			else:                
				shutil.rmtree(base_path + val_path + image_class)
			
		if not os.path.exists(base_path + val_path + image_class):
			os.makedirs(base_path + val_path + image_class)        
	
	# Move images
	#
	# Loop over all training classes
	for image_class in image_classes_list:
		
		# Loop over all images for the current image class
		image_name_list = next(os.walk(base_path + train_path + image_class))[2]                
		for image_name in image_name_list:
			
			# Move to val
			if np.random.rand() < val_fraction:
				file_name = base_path + train_path + image_class + "/" + image_name
				file_name_new = base_path + val_path + image_class + "/" + image_name
				
				shutil.move(file_name, file_name_new)   
				# print(file_name, file_name_new)
				
def prepare_image_data_generators(base_path, train_path, val_path, batch_size, img_width, img_height):
	""" Prepares keras data generators for train and validation
	
	Args:
		base_path: Base path of model
		train_path: Base train path under <base_path>
		val_path: Base val path under <base_path>
		batch_size: Batch size, e.g. 32
		img_width: Target width of image input to neural net after rescaling (299 for inceptionV3)
		img_height: Target height of image input to neural net after rescaling (299 for inceptionV3)
	
	Returns:
		train_generator (return of ImageDataGenerator.flow_from_directory)
		validation_generator (return of ImageDataGenerator.flow_from_directory)
	"""
	
	# Image augumentation configuration for training
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			rotation_range=45,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=True)

	# Image augumentation configuration for validation
	# only rescaling
	validation_datagen = ImageDataGenerator(rescale=1./255)

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
			base_path + train_path,  # this is the target directory
			target_size=(img_width, img_height),  # all images will be resized to img_width, img_height
			batch_size=batch_size,
			class_mode='categorical')

	# this is a similar generator, for validation data
	validation_generator = validation_datagen.flow_from_directory(
			base_path + val_path,
			target_size=(img_width, img_height),
			batch_size=batch_size,
			class_mode='categorical')	
	
	return (train_generator, validation_generator)
	
def get_image_classes(full_train_path):
	""" Determine labels of images classes from folders in <full_train_path>
	
	Args:
		full_train_path (string): path containing training data in subfolders
		
	Returns:
		image_classes_list (list of strings)	
	"""
	image_classes_list = []
	
	if (os.path.isdir(full_train_path)):
		image_classes_list = next(os.walk(full_train_path))[1]
		image_classes_list = sorted(image_classes_list)
	return (image_classes_list)
	
def count_class_weights(train_generator, image_classes_list, train_path, val_path):	
	""" Determine class weight parameters to compensate class imbalance during training
	
	Args:
		train_generator:
		image_classes_list:
		train_path:
		val_path:
		
	Returns:
		image_classes_list (list of strings)	
	"""	

	result_columns = ["x", "y", "filename"]
	path_dict = {"train" : train_path, "val" : val_path}
	class_weight = {}

	for current_type in path_dict:
		print(current_type)
		
		current_type_count = 0
		for image_class in image_classes_list:
			current_path = base_path + path_dict[current_type] + image_class    
			num_files_current_path = next(os.walk(current_path))[2]    
			result_columns.append(image_class)        
			current_type_count = current_type_count +  len(num_files_current_path)
			
			print("# class \'" + image_class + "\': " + str(len(num_files_current_path)))
			
			if current_type == "train":
				class_weight[train_generator.class_indices[image_class]] = len(num_files_current_path)
			
		print("----------------------------")
		print("Total " + current_type + ":", current_type_count, "\n")
		
	# Compute class weight to balance imbalanced training data

	total_count = 0
	for class_id in class_weight:
		total_count += class_weight[class_id]
		
	for class_id in class_weight:
		current_n = class_weight[class_id]
		class_weight[class_id] = total_count / current_n
		
	# The class weights multiplied by the number of samples should be equal for all classes
	print("Class weights: ", class_weight)

	print("Class indices", train_generator.class_indices)

	return(class_weight)
	
def	get_model(num_classes, img_width, img_height):
	""" Returns InceptionV3 based model, pretrained on image-net to be trained with num_classes
	
	Args:
		num_classes = number of classes
		img_width, img_height: 299 for InceptionV3
	Returns:
		Model object
	"""
	
	# Clean up Keras
	K.clear_session()
	input_shape = (img_width, img_height, 3)
	
	# Define base model (Inception V3, trained on image net, without top layers)
	image_net_base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
	
	# Define top model  
	input_tensor = Input(shape = input_shape)

	bn = BatchNormalization()(input_tensor)
	x = image_net_base_model(bn)
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	output = Dense(num_classes, activation='softmax')(x)

	model = Model(input_tensor, output)

	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.5e-4, momentum=0.9), metrics=['accuracy'])

	return (model)
	
	
def train_model(model, model_path, model_name, train_generator, validation_generator, batch_size, class_weight):
	""" Trains a model
		Args:
			model: Keras model object
			model_path: Path to save model
			model_name: Filename of model
			train_generator: Generator object for training images
			validation_generator: Generator object for validation images
			batch_size: How many images to feed into the network at once
			class_weight: List containing the relative frequencies of the classes (n_i/n_all)
			
			model_name, train_generator, validation_generator, batch_size, class_weight)
		Returns:
			model: trained model
			history: A keras history object
	"""

	callbacks = [ModelCheckpoint(model_path + model_name, monitor='val_acc', verbose=1, save_best_only=True),
				ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, cooldown=1, verbose=1, min_lr=1e-7)]
	history = model.fit_generator(train_generator,
								  steps_per_epoch = train_generator.n // batch_size,
								  epochs = n_epochs,
								  validation_data = validation_generator,
								  validation_steps = validation_generator.n // batch_size,
								  verbose = 1,
								  class_weight = class_weight,
								  callbacks = callbacks)

	return model, history
	

def save_learning_curves(history, results_base_path, model_name):
	""" Saves the learning curve from a given Keras history object
	Args:
		history: Keras history object
		results_base_path: Path to store the file
		model_name: Name of model
	Returns:
		Nothing
	"""	
	learning_curves = pd.DataFrame()
	learning_curves["acc"] = history.history["acc"]
	learning_curves["val_acc"] = history.history["val_acc"]
	learning_curves["loss"] = history.history["loss"]
	learning_curves["val_loss"] = history.history["val_loss"]	
	learning_curves.to_csv(results_base_path + model_name + "_learning.csv", index=False)
	
def main():
	# Get class labels
	image_classes_list = get_image_classes(base_path + train_path)

	if (len(image_classes_list) > 0):
		if do_val_split == True:
			split_validation_data(image_classes_list, base_path, train_path, val_path)
			
		# Get image data generators
		train_generator, validation_generator = prepare_image_data_generators(base_path, train_path, val_path, batch_size, img_width, img_height)
		
		# Determine class weight parameters to compensate class imbalance during training
		class_weight = count_class_weights(train_generator, image_classes_list, train_path, val_path)
		
		# Get InceptionV3 based model, pretrained on image-net to be trained with num_classes classes
		model = get_model(len(image_classes_list), img_width, img_height)
		
		# Train model
		model, history = train_model(model, model_path, model_name, train_generator, validation_generator, batch_size, class_weight)
		
		# Load model with best parameters on val
		model.load_weights(model_path + model_name)
		
		# Evaluation
		score = model.evaluate_generator(validation_generator, steps = 100)
		print(score[1])
		
		# Save learning curve
		save_learning_curves(history, results_base_path, model_name)		

	else:
		print("Training data folder \"%s\" not a directory or empty." % (base_path + train_path))
				
if __name__ == "__main__":
    main()