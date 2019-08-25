# USAGE
# python train_models.py --output output --models models

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from pyimagesearch.nn.conv import FCHeadNet
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-m", "--models", required=True,
	help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
	help="# of models to train")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the
# range [0, 1]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")

baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(32, 32, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(labelNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), epochs=1,
	steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
for layer in baseModel.layers[15:]:
	layer.trainable = True

# for the changes to the model to take affect we need to recompile
# # the model, this time using SGD with a *very* small learning rate
# print("[INFO] re-compiling model...")
# opt = SGD(lr=0.001)
# model.compile(loss="categorical_crossentropy", optimizer=opt,
# »·······metrics=["accuracy"])
#
# # train the model again, this time fine-tuning *both* the final set
# # of CONV layers along with our set of FC layers
# print("[INFO] fine-tuning model...")
# model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
# 	validation_data=(testX, testY), epochs=100,
# 	steps_per_epoch=len(trainX) // 32, verbose=1)
# loop over the number of models to train
for i in np.arange(0, args["num_models"]):
	# initialize the optimizer and model
	print("[INFO] training model {}/{}".format(i + 1,
		args["num_models"]))
	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#
	if i==1:
		opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9,
				nesterov=True)
	# if i ==3:
	# 	opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


	# model = MiniVGGNet.build(width=32, height=32, depth=3,
	# 	classes=10)

	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY), epochs=1,
		steps_per_epoch=len(trainX) // 64, verbose=1)

	# save the model to disk
	p = [args["models"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(p))

	# evaluate the network
	predictions = model.predict(testX, batch_size=64)
	report = classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=labelNames)

	# save the classification report to file
	p = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(p), "w")
	f.write(report)
	f.close()

	# plot the training loss and accuracy
	# p = [args["output"], "model_{}.png".format(i)]
	# plt.style.use("ggplot")
	# plt.figure()
	# plt.plot(np.arange(0, 40), H.history["loss"],
	# 	label="train_loss")
	# plt.plot(np.arange(0, 40), H.history["val_loss"],
	# 	label="val_loss")
	# plt.plot(np.arange(0, 40), H.history["acc"],
	# 	label="train_acc")
	# plt.plot(np.arange(0, 40), H.history["val_acc"],
	# 	label="val_acc")
	# plt.title("Training Loss and Accuracy for model {}".format(i))
	# plt.xlabel("Epoch #")
	# plt.ylabel("Loss/Accuracy")
	# plt.legend()
	# plt.savefig(os.path.sep.join(p))
	# plt.close()
