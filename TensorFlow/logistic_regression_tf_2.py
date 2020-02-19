''' Calculate prediction model 
    with Tensorflow v.2
'''

#Import neccesary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf

#Parameters
batch_size = 32
train_loss_results = []
train_accuracy_results = []
learning_rate = 0.03
num_epochs = 201

# Read data and define data frame
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)

# Define column and species
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data_feature_names = column_names[:-1]
labels_names = column_names[-1]

# Transform data set into a single tensor
train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, batch_size, column_names=column_names, label_name=labels_names, num_epochs=1)

# CPacking all vectors in a single tensor
def pack_features(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# Recompose dataset
train_dataset = train_dataset.map(pack_features)

# Create the model
prediction_model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # Number of inputs in this case data_feature
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# Loss function
loss_definition = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(model, x, y, training):
  y_ = model(x, training=training)
  return loss_definition(y_true=y, y_pred=y_)

# Gradient function and Optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

def gradient_function(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss_function(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Training Loop
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, gradient = gradient_function(prediction_model, x, y)
        optimizer.apply_gradients(zip(gradient, prediction_model.trainable_variables))
        # Track Pogress
        epoch_loss_avg(loss_value)
        epoch_accuracy(y, prediction_model(x, training=True))
        # Append Results
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))













