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
import sys
import numpy as np
from skimage import data, io, filters, transform
from skimage.transform import resize
import pandas as pd
import shutil
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import load_model
from keras import backend as K
from keras import applications
from keras import optimizers
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Set parameters for input and output of the prediction below
"""

# Base path
base_path = "./"

# Path of output images
predict_base_path = base_path + "prediction_data/"

# Path of DL model (.h5)
model_name = base_path + "model_data/dataset - V4/model/Inflammation_Inception_V3.h5"

# Path of result data
results_base_path = base_path + "prediction data/"

# Image dimensions (299x299 for InceptionV3 based nets)
img_width, img_height = 299, 299

# Number of classes
num_classes = 5

def prepare_predict_list(predict_path, subfolder_str = "/images"):
	""" Create data frame for prediction data containing png image tiles
		Note: assumes image name in format "sampleId_secondarySampleId_x_y.png"
	
	Args:
		Predict path (string): path containing the image location
		subfolder_str: Subfolder name containing the images
	Returns:
		Pandas DataFrame: List of images to predict with columns for sample, x, y and filename
	"""		
	predict_list = pd.DataFrame(columns={"sample", "x", "y", "filename"})

	# Change order of columns
	predict_list = predict_list[["sample", "x", "y", "filename"]]
		
	image_path = predict_path + subfolder_str
	
	if (os.path.exists(image_path)):
		filenames = next(os.walk(image_path))[2]		
		for file in filenames:
			if file[-4:] == ".png":
				sample = file.split("_")[0] + "_" + file.split("_")[1]
				y = int(file.split("_")[3].split(".")[0])
				x = int(file.split("_")[2])
				predict_list = predict_list.append({"y" : y, "x" : x, "filename" : file, "sample" : sample}, ignore_index=True)
				# print(file,sample,x,y)
	else:
		print ("Folder \"%s\" not found!" % (image_path))

	return predict_list
	
def get_predict_generator(predict_path):
	""" Get the Keras predict generator
	Args:
		Predict path (string): path containing the image location
	
	Return:
		predict_datagen object
	"""
	# 	
	predict_datagen = ImageDataGenerator(rescale=1./255)

	# Predict generator
	predict_generator = predict_datagen.flow_from_directory(
			predict_path,
			target_size=(img_width, img_height),
			batch_size = 1,
			class_mode=None,
			shuffle=False)
			
	return predict_generator

def get_trained_model(predict_path, model_name):
	""" Prepare CNN for prediction and get trained model
	
	Args:
		Predict path (string): path containing the image location
		Model_name (string): Filename of the model to load
	Returns:
		Model object with loaded weights
	"""
	
	# Clean up Keras
	K.clear_session()
			
	# Load trained CNN based on Inception V3
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

	model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

	# Load weights of pre-trained model
	model.load_weights(model_name)
	
	return model

def predict(model, predict_generator):
	""" Predict
	Args:
		model: Model object with loaded weights	
		predict_datagen		
	Returns:
		prediction_result (numpy array)
		
	"""
	prediction_result = model.predict_generator(predict_generator, predict_generator.n, verbose=1)
	
	return prediction_result

def get_processed_prediction(prediction_result, predict_list, predict_generator):
	""" Will renormalize the prediction (without ignore classes)
	    and create a single score
	Args:
		prediction_result
		predict_list (Pandas DataFrame): List of images to predict with columns for sample, x, y and filename
		predict_generator object
	Return:
		predict_list (Pandas DataFrame)
	"""
	
	# Add column to resulting output dataframe
	predict_list["inflammation"] = np.NaN
		
	i = 0
	for name_str in predict_generator.filenames:		
			
		p_inflamed_0 = prediction_result[i][0]
		p_inflamed_1 = prediction_result[i][1]
		p_inflamed_2 = prediction_result[i][2]
		p_inflamed_3 = prediction_result[i][3]
		p_ignore = prediction_result[i][4]	
		
		if p_ignore == np.max(prediction_result[i][:]):
			inflammation = np.NaN
		else:       
			# Compute inflammation score
			# Re-normalize to 1 without p_ignore
			numerator = np.sum(prediction_result[i][:])        
			denominator = np.sum(prediction_result[i][:-1])
			
			p_inflamed_0 = p_inflamed_0*numerator/denominator
			p_inflamed_1 = p_inflamed_1*numerator/denominator
			p_inflamed_2 = p_inflamed_2*numerator/denominator
			p_inflamed_3 = p_inflamed_3*numerator/denominator
			
			# Weighted inflammation score
			inflammation = p_inflamed_0*0 + p_inflamed_1*1 + p_inflamed_2*2 + p_inflamed_3*3
			
		file_name = name_str.split("/")[1]				
		
		predict_list.set_value(predict_list[predict_list["filename"] == file_name].index[0], "Ignore", p_ignore)
		predict_list.set_value(predict_list[predict_list["filename"] == file_name].index[0], "inflammation", inflammation)
		
		i = i + 1 
		
	return predict_list

def get_summary_result(predict_list):
	""" Create a summary of the result file:
		Args:
			predict_list (Pandas DataFrame): Result object 
		Return:
			summary_prediction (Pandas Dataframe)
	"""
	summary_prediction = pd.DataFrame(columns={"sample", "mean_inflammation"})

	for sample in np.unique(predict_list["sample"]):
		# Create dataframe of current experiment
		current_prediction = predict_list[predict_list["sample"] == sample]
		
		summary_prediction = summary_prediction.append({"sample" : sample, "mean_inflammation": current_prediction["inflammation"].mean()}, ignore_index=True)    

	summary_prediction = summary_prediction[["sample", "mean_inflammation"]]    
	return summary_prediction	
	
	

def main():
	if (len(sys.argv) > 1):
		experiment_name = str(sys.argv[1])
		print("Path containing images for prediction: %s." % (predict_base_path + experiment_name))
		
		# Get list of items to predict	
		predict_list = prepare_predict_list(predict_base_path + experiment_name)
		print("%d images for prediction." % (predict_list.shape[0]))
		
		if (predict_list.shape[0]>0):
			# Prepare network and get model object
			model = get_trained_model(predict_base_path + experiment_name, model_name)
			print("Model %s loaded." % (model_name))
			
			# Prepare generator object to deliver the images to the cnn
			predict_generator = get_predict_generator(predict_base_path + experiment_name)
			
			# Predict
			prediction_result = predict(model, predict_generator)	
			
			# Renormalize the result (without ignore class) and compute a single score
			predict_list = get_processed_prediction(prediction_result, predict_list, predict_generator)		
			
			# Save result (full details)
			predict_list.to_csv(results_base_path + experiment_name + "/" + experiment_name + "_inflammation_score_full_results.csv", index = False)
			print("Result saved to %s." % (results_base_path + experiment_name + "/" + experiment_name + "_inflammation_score_full_results.csv"))
			
			# Get summary_prediction
			summary_prediction = get_summary_result(predict_list)
			
			# Save summary result
			summary_prediction.to_csv(results_base_path + experiment_name + "/" + experiment_name + "_inflammation_score_summary.csv", index = False, sep = ";")
			print("Summary result saved to %s." % (results_base_path + experiment_name + "/" + experiment_name + "_inflammation_score_summary.csv"))
		
	else:
		print("Experiment name corresponding to a folder (containing ./images/ subfolder with images) must be given as argument")

if __name__ == "__main__":
    main()