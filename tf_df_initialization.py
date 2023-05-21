from os.path import dirname
import tensorflow_decision_forests as tfdf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from IPython.core.magic import register_line_magic
from IPython.display import Javascript
from wurlitzer import sys_pipes

#Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)

#Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("/tmp/penguins.csv")

#Display the first 3 examples
print(dataset_df.head(3))

#Encode the categorical label into an integer
#
#Details:
#This stage is necessary if your classification label is represented as a
#string. Note: Keras expected classification labels to be integers.

#Name of the label column.
label = "species"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

#Split the dataset into a training and a testing dataset

def split_dataset(dataset, test_ratio=0.30):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

set_cell_height = 300

#Specify the model
model_1 = tfdf.keras.RandomForestModel()

#Optionally, add evaluation metrics
model_1.compile(metrics=["accuracy"])

#Train the model
#"sys_pipes" is optional. It enables the display of the training logs
with sys_pipes():
    model_1.fit(x=train_ds)

evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

model_1.save("/tmp/my_saved_model")

tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)

set_cell_height = 300
print(model_1.summary())

#The input features
print(model_1.make_inspector().features())

#The feature importances
print(model_1.make_inspector().variable_importances())

print(model_1.make_inspector().evaluation())

set_cell_height = 150
print(model_1.make_inspector().training_logs())

import matplotlib.pyplot as plt

logs = model_1.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")
plt.show()

model_1.make_inspector().export_to_tensorboard("/tmp/tensorboard_logs")

print(tfdf.keras.get_all_models())

#help works anywhere
print(help(tfdf.keras.RandomForestModel))

print(tfdf.keras.RandomForestModel)

feature_1 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_2 = tfdf.keras.FeatureUsage(name="island")

all_features = [feature_1, feature_2]

#Note: This model is only trained with two features. It will not be as good as
#the one trained on all features.

model_2 = tfdf.keras.GradientBoostedTreesModel(
    features=all_features, exclude_non_specified_features=True)

model_2.compile(metrics=["accuracy"])
model_2.fit(x=train_ds, validation_data=test_ds)

print(model_2.evaluate(test_ds, return_dict=True))

set_cell_height = 300

feature_1 = tfdf.keras.FeatureUsage(name="year", semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
feature_2 = tfdf.keras.FeatureUsage(name="bill_length_mm")
feature_3 = tfdf.keras.FeatureUsage(name="sex")
all_features = [feature_1, feature_2, feature_3]

model_3 = tfdf.keras.GradientBoostedTreesModel(features=all_features, exclude_non_specified_features=True)
model_3.compile(metrics=["accuracy"])

with sys_pipes():
    model_3.fit(x=train_ds, validation_data=test_ds)

#A classical but slightly more complex model
model_6 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=8)
model_6.fit(x=train_ds)

#A more complex, but possibly, more accurate model
model_7 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500,
    growing_strategy="BEST_FIRST_GLOBAL",
    max_depth=8,
    split_axis="SPARSE_OBLIQUE",
    categorical_algorithm="RANDOM",
)
model_7.fit(train_ds)

#A good template of hyper-parameters
model_8 = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
model_8.fit(x=train_ds)

#The hyper-parameter templates of the Gradient Boosted Tree model
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())

set_cell_height = 300

body_mass_g = tf.keras.layers.Input(shape=(1, ), name="body_mass_g")
body_mass_kg = body_mass_g / 1000.0

bill_length_mm = tf.keras.layers.Input(shape=(1, ), name="bill_length_mm")

raw_inputs = {"body_mass_g": body_mass_g, "bill_length_mm": bill_length_mm}
processed_inputs = {"body_mass_kg": body_mass_kg, "bill_length_mm": bill_length_mm}

#"preprocessor" contains the preprocessing logic
preprocessor = tf.keras.Model(inputs=raw_inputs, outputs=processed_inputs)

#"model_4" contains both the pre-processing logic and the decision forest.
model_4 = tfdf.keras.RandomForestModel(preprocessing=preprocessor)
model_4.fit(x=train_ds)

print(model_4.summary())

def g_to_kg(x):
    return x / 1000

feature_columns = [
    tf.feature_column.numeric_column("body_mass_g", normalizer_fn=g_to_kg),
    tf.feature_column.numeric_column("bill_length_mm"),
]

preprocessing = tf.keras.layers.DenseFeatures(feature_columns)

model_5 = tfdf.keras.RandomForestModel(preprocessing=preprocessing)
model_5.compile(metrics=["accuracy"])
model_5.fit(x=train_ds)

dataset_df = pd.read_csv("/tmp/abalone.csv")
print(dataset_df.head(3))

#Split the dataset into a training and testing dataset
train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing".format(
    len(train_ds_pd), len(test_ds_pd)))

#Name of the label column
label = "Rings"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

set_cell_height = 300

#Configure the model
model_7 = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)

#Optional
model_7.compile(metrics=["mse"])

#Train the model
with sys_pipes():
    model_7.fit(x=train_ds)

#Evaluate the model on the test dataset
evaluation = model_7.evaluate(test_ds, return_dict=True)

print(evaluation)
print()
print(f"MSE: {evaluation['mse']}")
print(f"RMSE: {math.sqrt(evaluation['mse'])}")

set_cell_height = 200

archive_path = tf.keras.utils.get_file("letor.zip",
                                       "https://download.microsoft.com/download/E/7/E/E7EABEF1-4C7B-4E31-ACE5-73927950ED5E/Letor.zip",
                                       extract=True)

#Path to the train and test dataset using libsvm format.
raw_dataset_path = os.path.join(dirname(archive_path), "OHSUMED/Data/All/OHSUMED.txt")

def convert_libsvm_to_csv(src_path, dst_path):
    dst_handle = open(dst_path, "w")
    first_line = True
    for src_line in open(src_path, "r"):
        #Note: The last 3 items are comments
        items = src_line.split(" ")[:-3]
        relevance = items[0]
        group = items[1].split(":")[1]
        features = [item.split(":") for item in items[2:]]

        if first_line:
            #Csv header
            dst_handle.write("relevance,group," ",".join(["f" + feature[0] for feature in features]) + "\n")
            first_line = False
        dst_handle.write(relevance + ",g_" + group + "," + (",".join([feature[1] for feature in features])) + "\n")
    dst_handle.close()

#Convert the dataset
csv_dataset_path = "/tmp/ohsumed.csv"
convert_libsvm_to_csv(raw_dataset_path, csv_dataset_path)

#Load a dataset into a Pandas DataFrame
dataset_df = pd.read_csv(csv_dataset_path)

#Display the first 3 examples
print(dataset_df.head())

train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

#Display the first 3 examples of the training dataset
print(train_ds_pd.head(3))

#Name of the relevance and grouping columns
relevance = "relevance"

ranking_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=relevance, task=tfdf.keras.Task.RANKING)
ranking_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=relevance, task=tfdf.keras.Task.RANKING)

set_cell_height = 400

model_8 = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.RANKING,
    ranking_group="group",
    num_trees=50)

with sys_pipes():
    model_8.fit(x=ranking_train_ds)

set_cell_height = 400
print(model_8.summary())