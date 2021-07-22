import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout,
GlobalAveragePooling2D, MaxPooling2D,MaxPool2D, Flatten

#loading the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
zoom_range=0.2, width_shift_range=0.2,
height_shift_range=0.2,
fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

base= 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
train_set = train_datagen.flow_from_directory(base+'/train', target_size=(224, 224),
batch_size = 32, class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(base+'/valid', target_size=(224,224),
batch_size=32, class_mode='categorical')

#CNN neural network
base_model = tf.keras.applications.MobileNet(weights='imagenet',
                                             input_shape = (224,224,3), include_top=False)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(38)(x)

mobilenet_model = tf.keras.Model(inputs, outputs, name='pretrained_mobilenet')
mobilenet_model.summary()
mobilenet_model.compile(optimizer=tf.keras.optimizers.Adam(),
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
cbac = tf.keras.callbacks.ModelCheckpoint(
"first_train.h5",
monitor="val_loss",
verbose=1,
save_best_only=True)
epochs = 15

results = mobilenet_model.evaluate(valid_set)
print('val loss:', results[0])
print('val acc:', results[1])

base_model.trainable = True
mobilenet_model.summary()

mobilenet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
cbac = tf.keras.callbacks.ModelCheckpoint(
"second_train.h5",
monitor="val_loss",
verbose=1,
save_best_only=True)

epochs = 5

mobilenet_history_ft = mobilenet_model.fit(train_set, steps_per_epoch=150,
validation_data=valid_set,
epochs=epochs,
validation_steps=100,callbacks=[cbac])

results2 = mobilenet_model.evaluate(valid_set)
print('val loss:', results2[0])
print('val acc:', results2[1])

#testing the trained model
model = load_model("second_train.h5")
test_im,test_lab = next(valid_set)
y_pred = []
y_test = []
for i,j in zip(model.predict(test_im),test_lab):
y_pred.append(i.argmax())
y_test.append(j.argmax())

print("accuracy score: ",accuracy_score(y_test,y_pred)*100,"%")

classes = os.listdir(r"New Plant Diseases Dataset(Augmented)\New Plant Diseases
Dataset(Augmented)\train")
im_add = "test_upload_images/test"
plt.figure(figsize=(16,10))
for j,i in enumerate(os.listdir(im_add)[:16]):
img = cv2.imread(os.path.join(im_add,i))
img = img/255
img = cv2.resize(img,(224, 224))
img1 = img.reshape((1, 224, 224, 3))
type_pred = model.predict(img1)
plt.subplot(4,4,j+1)
plt.imshow(img)
plt.title(f'actual: {i.strip(".JGP")}\npred: {classes[type_pred.argmax()]}')

plt.tight_layout()
plt.show()                     
