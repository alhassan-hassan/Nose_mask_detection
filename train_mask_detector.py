# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

init_learningRate = 1e-4    # initializing the learning rate.
epochs = 20      # number of epochs to train for
batch_size = 32          # the batch_size specification

DIRECTORY = r"C:\Users\User\Desktop\Linear Algebra\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("Loading images...")

data_for_images = []        # a list object to store the images
labels_for_image_folders = []      # a list for storing the categories of the images

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)         # join the root directory to the images folders
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))       # specifying the image size to store
        image = img_to_array(image)
        image = preprocess_input(image)

        data_for_images.append(image)
        labels_for_image_folders.append(category)

# performing encoding on the labels
label_bin = LabelBinarizer()
labels_for_image_folders = label_bin.fit_transform(labels_for_image_folders)
labels_for_image_folders = to_categorical(labels_for_image_folders)

data_for_images = np.array(data_for_images, dtype="float32")  # converting the images to numpy categories(mostly zeroes and ones)
labels_for_image_folders = np.array(labels_for_image_folders)

(trainX, testX, trainY, testY) = train_test_split(
    data_for_images, labels_for_image_folders, test_size=0.20, stratify=labels_for_image_folders, random_state=42)

# construct the training image generator for data augmentation
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel_trained = MobileNetV2(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
head_model = baseModel_trained.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# placing the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel_trained.input, outputs=head_model)

# freeze all the layers of the base model to prevent it from updating
for layer in baseModel_trained.layers:
    layer.trainable = False

# compile our model
print("Compiling the model...")
opt = Adam(lr=init_learningRate, decay=init_learningRate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("Training the head...")
H = model.fit(
    augmentation.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)

# make predictions on the testing set
print("Evaluating the network...")
predict_Index = model.predict(testX, batch_size=batch_size)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predict_Index = np.argmax(predict_Index, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predict_Index,
                            target_names=label_bin.classes_))

# serialize the model to disk
print("Saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plotting the loss and accuracy training status
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
