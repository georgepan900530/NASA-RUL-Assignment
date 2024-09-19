import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU
from torchinfo import summary
import keras
import seaborn as sns
import json
import os


with open("config.json", "r") as f:
    config = json.load(f)

PM_train = config["train_path"]
PM_test = config["test_path"]
PM_truth = config["GT_path"]

# Read training data - Aircraft engine run-to-failure data
train_df = pd.read_csv(
    PM_train, sep=" ", header=None
)  # Read the txt file, use appropriate separator and header
train_df.drop(
    [26, 27], axis=1, inplace=True
)  # Explore the data on your own and remove unnecessary columns
train_df.columns = (
    ["id", "cycle"]
    + [f"settings{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)  # Assign names to all the columns

train_df = train_df.sort_values(["id", "cycle"])  # Sort by id and cycle

# Read test data - Aircraft engine operating data without failure events recorded
test_df = pd.read_csv(
    PM_test, sep=" ", header=None
)  # Read the txt file, use appropriate separator and header
test_df.drop(
    [26, 27], axis=1, inplace=True
)  # Explore the data on your own and remove unnecessary columns
test_df.columns = (
    ["id", "cycle"]
    + [f"settings{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)  # Assign names to all the columns

# Read ground truth data - True remaining cycles for each engine in testing data
truth_df = pd.read_csv(
    PM_truth, sep=" ", header=None
)  # Read the txt file, use appropriate separator and header
truth_df.drop(
    [1], axis=1, inplace=True
)  # Explore the data on your own and remove unnecessary columns

print(train_df.describe().T)

columns2drop = [
    "settings1",
    "settings2",
    "settings3",
    "sensor1",
    "sensor10",
    "sensor18",
    "sensor19",
    "sensor5",
    "sensor6",
    "sensor16",
]
train_df.drop(columns2drop, axis=1, inplace=True)

test_df.drop(columns2drop, axis=1, inplace=True)


#######
# TRAIN
#######
# Data Labeling - generate column RUL (Remaining Useful Life or Time to Failure)

# TODO: Calculate the maximum cycle value for each engine (id) and store it in a new DataFrame (rul)
rul = train_df.groupby(["id"]).max().reset_index()
rul = rul[["id", "cycle"]]
# TODO: Rename the columns in the rul DataFrame
rul.columns = ["id", "max_cycle"]
# TODO: Merge the rul DataFrame with the original train_df based on the 'id' column
train_df = pd.merge(train_df, rul, on="id", how="left")
# TODO: Calculate the Remaining Useful Life (RUL) by subtracting the current cycle from the maximum cycle
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
# TODO: Remove the temporary column used to calculate RUL
train_df.drop(["max_cycle"], axis=1, inplace=True)

# Generate label columns for training data
# We will only make use of "label1" for binary classification,
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
w1 = 30
w0 = 15

# TODO: Create a binary label ('label1') indicating if the engine will fail within w1 cycles (1) or not (0)
train_df["label1"] = (train_df["RUL"] <= w1).astype(
    int
)  # Replace with the correct threshold value and label values
# TODO: Initialize a second label ('label2') as a copy of 'label1'
train_df["label2"] = train_df["label1"]
# TODO: Update 'label2' to indicate if the engine will fail within w0 cycles (2) or not (0/1)
train_df.loc[train_df["RUL"] <= w0, "label2"] = (
    2  # Replace with the correct threshold value and label value
)


# MinMax normalization (from 0 to 1)
# TODO: Create a normalized version of the 'cycle' column (e.g., 'cycle_norm') using the original 'cycle' values
train_df["cycle_norm"] = (train_df["cycle"] - train_df["cycle"].min()) / (
    train_df["cycle"].max() - train_df["cycle"].min()
)  # Replace with the correct normalization code
# TODO: Select the columns to be normalized (all columns except 'id', 'cycle', 'RUL', 'label1', and 'label2')
cols_normalize = train_df.columns.difference(
    ["id", "cycle", "RUL", "label1", "label2"]
)  # Replace with the correct column selection code
# TODO: Initialize a MinMaxScaler object to scale values between 0 and 1
min_max_scaler = (
    preprocessing.MinMaxScaler()
)  # Replace with the correct scaler initialization code
# TODO: Apply MinMaxScaler to the selected columns and create a new normalized DataFrame
norm_train_df = pd.DataFrame(
    min_max_scaler.fit_transform(train_df[cols_normalize]),
    columns=cols_normalize,
    index=train_df.index,
)  # Replace with the correct normalization code
# TODO: Join the normalized DataFrame with the original DataFrame (excluding normalized columns)
join_df = train_df[["id", "cycle", "RUL", "label1", "label2"]].join(
    norm_train_df
)  # Replace with the correct join code
# TODO: Reorder the columns in the joined DataFrame to match the original order
train_df = join_df.reindex(
    columns=train_df.columns
)  # Replace with the correct reindexing code

######
# TEST
######
# MinMax normalization (from 0 to 1)
# TODO: Similar to the MinMax normalization done for Train, complete the code below.
test_df["cycle_norm"] = (test_df["cycle"] - test_df["cycle"].min()) / (
    test_df["cycle"].max() - test_df["cycle"].min()
)
test_cols_normalize = test_df.columns.difference(["id", "cycle"])
norm_test_df = pd.DataFrame(
    min_max_scaler.fit_transform(test_df[test_cols_normalize]),
    columns=test_cols_normalize,
    index=test_df.index,
)
test_join_df = test_df[["id", "cycle"]].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)

# We use the ground truth dataset to generate labels for the test data.
# generate column max for test data
# TODO: Calculate the maximum cycle value for each engine (id) in the test data and store it in a new DataFrame (rul)
rul = test_df.groupby(["id"]).max().reset_index()
rul = rul[["id", "cycle"]]
# TODO: Rename the columns in the rul DataFrame
rul.columns = ["id", "max"]
# TODO: Merge the rul DataFrame with the original test_df based on the 'id' column
test_df = pd.merge(test_df, rul, on="id", how="left")
truth_df.columns = ["more"]
truth_df["id"] = truth_df.index + 1
truth_df["max_truth"] = rul["max"] + truth_df["more"]
# TODO: Remove the temporary column used to calculate RUL
truth_df.drop(["more"], axis=1, inplace=True)

# TODO: Merge the adjusted truth_df with the test_df to generate RUL values for test data
test_df = pd.merge(test_df, truth_df[["id", "max_truth"]], on="id", how="left")
# TODO: Calculate the Remaining Useful Life (RUL) by subtracting the current cycle from the maximum cycle
test_df["RUL"] = test_df["max_truth"] - test_df["cycle"]
# TODO: Remove the temporary column used to calculate RUL
test_df.drop(["max", "max_truth"], axis=1, inplace=True)

# Generate binary label columns (label1 and label2) based on RUL values and thresholds w0 and w1
# TODO: Similar to what you did in the train dataframe
test_df["label1"] = (test_df["RUL"] <= w1).astype(int)
test_df["label2"] = test_df["label1"]
test_df.loc[test_df["RUL"] <= w0, "label2"] = 2

train_df = train_df.reindex(
    columns=test_df.columns
)  # reorder the train_df so that it matches the order of test_df

# Draw the correlation matrix of training dataframe
corr = train_df.corr()
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(230, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
fig = sns.heatmap(
    corr,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
).get_figure()
fig_name = "correlation_matrix_of_training_df.png"
if not os.path.exists(config["save_fig_folder"]):
    os.mkdir(config["save_fig_folder"])
fig.savefig(os.path.join(config["save_fig_folder"], fig_name))

# TODO: Define window size and sequence length
sequence_length = 128  # Replace with the desired sequence length


# Function to reshape features into (samples, time steps, features) --> time steps is the same as sequence_length
# Note that this function only generate sequences for a single engine
def generate_sequences(id_df, sequence_length, feature_columns):
    """Generate sequences from a dataframe for a given id.
    Sequences that are under the sequence length will be considered.
    We can also pad the sequences in order to use shorter ones."""
    data_matrix = id_df[feature_columns].values
    num_elements = data_matrix.shape[0]

    for start, end in zip(
        range(0, num_elements - sequence_length), range(sequence_length, num_elements)
    ):
        yield data_matrix[
            start:end, :
        ]  # TODO: Replace with the correct code to yield sequences of feature values


# TODO: Select feature columns for sequence generation (e.g., sensor readings, settings)
sensor_columns = [
    c for c in train_df.columns if c.startswith("sensor")
]  # TODO: Replace with the correct list of sensor column names
sequence_columns = sensor_columns  # TODO: Replace with the correct list of sequence column names (including settings and sensors)

# TODO: Generate sequences for all engine ids in the training data
sequence_generator = []  # TODO: Replace with the correct code to generate sequences
for engine_id in train_df["id"].unique():
    sub_train_df = train_df[train_df["id"] == engine_id]
    # Generate sequences for a single engine
    sequences = generate_sequences(sub_train_df, sequence_length, sequence_columns)
    for sequence in sequences:
        sequence_generator.append(sequence)

# TODO: Convert generated sequences to a numpy array for LSTM input
sequence_array = np.array(
    sequence_generator
)  # TODO: Replace with the correct code to convert sequences to numpy array

print(sequence_array.shape)


# TODO: Function to generate labels -> (samples, time steps)
def generate_labels(id_df, sequence_length, label_column):
    """Generate labels for a given id."""
    data_matrix = id_df[label_column].values
    num_elements = data_matrix.shape[0]
    labels = [data_matrix[end] for end in range(sequence_length, num_elements)]
    labels = np.array(labels)
    return labels  # TODO: Replace with the correct code to generate labels


# TODO: Generate labels for all engine ids in the training data
label_generator = (
    []
)  # TODO: Replace with the correct code to generate labels for all engine ids
for engine_id in train_df["id"].unique():
    sub_train_df = train_df[train_df["id"] == engine_id]
    # Generate sequences for a single engine
    labels = generate_labels(sub_train_df, sequence_length, ["label1"])
    for label in labels:
        label_generator.append(label)
label_array = np.array(
    label_generator
)  # TODO: Replace with the correct code to convert labels to a numpy array
print(label_array.shape)

# Define the number of features and output units
nb_features = sequence_array.shape[2]
nb_out = label_array.shape[1]

# Create a Sequential model
model_lstm = Sequential()

# TODO: Add LSTM layers and Dropout layers to the model
# Note: Limit the total number of model parameters to 10,000
# Your code here:

model_lstm.add(
    LSTM(
        8, return_sequences=False, input_shape=(sequence_length, len(sequence_columns))
    )
)
# model_lstm.add(Dropout(0.5))
# model_lstm.add(LSTM(8, return_sequences=False))
# model_lstm.add(Dropout(0.5))

# Add a Dense output layer with sigmoid activation
model_lstm.add(Dense(units=nb_out, activation="sigmoid"))

# Compile the model with binary crossentropy loss and Adam optimizer
model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# TODO: Print the model summary
print(model_lstm.summary())

# TODO: Fit the network to the training data
checkpoints_folder = config["checkpoints_folder"]
if not os.path.exists(checkpoints_folder):
    print(f"Creating {checkpoints_folder}")
    os.mkdir(checkpoints_folder)
lstm_name = "lstm_best.keras"
history = model_lstm.fit(
    sequence_array,
    label_array,
    epochs=config["epochs"],  # TODO: Replace with the desired number of training epochs
    batch_size=config["batch_size"],  # TODO: Replace with the desired batch size
    validation_split=config[
        "validation_split"
    ],  # TODO: Replace with the desired validation split proportion
    verbose="auto",  # TODO: Replace with the desired verbosity level
    callbacks=[
        # TODO: Early stopping callback to stop training when validation loss stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00001,  # TODO: Replace with the minimum change in validation loss to qualify as improvement
            patience=15,  # TODO: Replace with the number of epochs to wait before stopping training
            verbose=1,  # TODO: Replace with the desired verbosity level
            mode="min",
        ),
        # TODO: Model checkpoint callback to save the best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoints_folder, lstm_name
            ),  # TODO: Replace with the file path to save the best model
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,  # TODO: Replace with the desired verbosity level
        ),
    ],
)

# TODO: summarize history for Accuracy
# TODO: Plot the training & validation accuracy over epochs and display the plot
# TODO: Save the plot to a file
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("LSTM Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()


# TODO: summarize history for Loss
# TODO: Plot the training & validation loss over epochs and display the plot
# TODO: Save the plot to a file
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
fig_name = "LSTM_Validation_vs_Training.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()


# TODO: Use the evaluate method to calculate the accuracy of the model on the training data
scores = model_lstm.evaluate(
    sequence_array, label_array
)  # TODO: Replace with the correct code to evaluate the model on the training data

# Print the accuracy of the model on the training data
print(f"Training Accuracy: {scores[1]}")

# make predictions and compute confusion matrix
# TODO: Use the predict method to make predictions on the training data
# TODO: Convert the predicted probabilities to class labels (e.g., using a threshold of 0.5)
y_pred = model_lstm.predict(
    sequence_array
)  # TODO: Use predict and convert probabilities to class labels
y_pred = (y_pred > 0.5).astype(int)
y_true = label_array

# TODO: Create a Pandas DataFrame from the predicted labels and save it to a CSV file
test_set = pd.DataFrame(
    y_pred, columns=["predicted_labels"]
)  # TODO: Replace with the correct code to create a DataFrame from the predicted labels
lstm_predictions = "LSTM"
lstm_full_path = os.path.join(
    config["save_csv_folder"], lstm_predictions, "lstm_predicted_labels_train.csv"
)
test_set.to_csv(
    lstm_full_path,
    index=False,
)

print("Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels")
# TODO: Compute the confusion matrix using confusion_matrix from sklearn.metrics
cm = confusion_matrix(
    y_true, y_pred
)  # TODO: Replace with the correct code to compute the confusion matrix
# print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="g", cmap="Blues", cbar=False
)  # fmt='g' to display integer values
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix of LSTM")
fig_name = "LSTM_Confusion_Matrix.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()
# TODO: Calculate the precision using precision_score and recall using recall_score from sklearn.metrics
precision = precision_score(
    y_true, y_pred
)  # TODO: Replace with the correct code to calculate precision
recall = recall_score(
    y_true, y_pred
)  # TODO: Replace with the correct code to calculate recall
print("precision = ", precision, "\n", "recall = ", recall)

# TODO: Pick the last sequence for each id in the test data
seq_array_test_last = []  # Replace with code to select last sequence for each id
for engine_id in test_df["id"].unique():
    sub_test_df = test_df[test_df["id"] == engine_id]

    last_sequence = sub_test_df[sequence_columns].values[-sequence_length:]
    if last_sequence.shape[0] >= sequence_length:
        seq_array_test_last.append(last_sequence)

# TODO: Convert to numpy array and ensure float32 data type
seq_array_test_last = np.array(seq_array_test_last).astype(np.float32)

# TODO: Pick the labels for the selected sequences
y_mask = (
    test_df.groupby("id").size() >= sequence_length
)  # TODO: Replace with code to select labels for sequences with length >= sequence_length
label_array_test_last = (
    []
)  # TODO: Replace with code to select labels for the selected sequences
for engine_id in test_df["id"].unique():
    if y_mask[engine_id]:  # Only consider engines with length >= sequence_length
        # Filter the data for the current engine_id
        sub_test_df = test_df[test_df["id"] == engine_id]

        # Select the label for the last sequence
        last_label = sub_test_df["label1"].values[-1]

        # Append the last label to the list
        label_array_test_last.append(last_label)

# Reshape and ensure float32 data type
label_array_test_last = np.array(label_array_test_last)
label_array_test_last = label_array_test_last.reshape(
    label_array_test_last.shape[0], 1
).astype(np.float32)


# TODO: Load the saved model if it exists
model_path = os.path.join(checkpoints_folder, lstm_name)
if os.path.isfile(model_path):
    estimator = load_model(
        model_path
    )  # TODO: Replace with code to load the saved model

# TODO: Evaluate the model on the test data
scores_test_lstm = estimator.evaluate(
    seq_array_test_last, label_array_test_last
)  # TODO: Replace with code to evaluate the model on the test data
print("Accuracy: {}".format(scores_test_lstm[1]))

# TODO: Make predictions and compute confusion matrix
y_pred_test = estimator.predict(
    seq_array_test_last
)  # TODO: Replace with code to make predictions and convert to class labels
y_pred_test = (y_pred_test > 0.5).astype(int)
y_true_test = label_array_test_last

# TODO: Create pandas dataframe of y_pred_test and save predictions to CSV file
test_set = pd.DataFrame(y_pred, columns=["predicted_labels"])
lstm_full_path_test = os.path.join(
    config["save_csv_folder"], lstm_predictions, "predicted_labels_test_lstm.csv"
)
test_set.to_csv(
    lstm_full_path_test,
    index=False,
)

# TODO: Compute confusion matrix
print("Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels")
cm = confusion_matrix(
    y_true_test, y_pred_test
)  # TODO: Replace with the correct code to compute the confusion matrix
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="g", cmap="Blues", cbar=False
)  # fmt='g' to display integer values
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix of LSTM (test)")
fig_name = "LSTM_Confusion_Matrix_testing.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()

# TODO: Compute precision, recall, and F1-score
precision_test_lstm = precision_score(
    y_true_test, y_pred_test
)  # TODO: Replace with code to compute precision
recall_test_lstm = recall_score(
    y_true_test, y_pred_test
)  # TODO: Replace with code to compute recall
f1_test_lstm = (
    2
    * precision_test_lstm
    * recall_test_lstm
    / (precision_test_lstm + recall_test_lstm)
)  # TODO: Replace with code to compute F1-score
print(
    "Precision: ",
    precision_test_lstm,
    "\n",
    "Recall: ",
    recall_test_lstm,
    "\n",
    "F1-score:",
    f1_test_lstm,
)

# TODO: Plot predicted and actual data for visual verification
plt.figure(figsize=(10, 6))

# Plot actual values (ground truth)
plt.plot(label_array_test_last, "bo", label="Actual Data", markersize=4, linestyle="")

# Plot predicted values
plt.plot(y_pred_test, "rx", label="Predicted Data", markersize=4, linestyle="")

plt.title("Predicted vs Actual Data")
plt.xlabel("Sample Index")
plt.ylabel("Binary Value")
plt.legend()
fig_name = "LSTM_Predicted_vs_Actual_test.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()

# Define the number of features and output units
nb_features = sequence_array.shape[2]
nb_out = label_array.shape[1]

# Create a Sequential model
model_gru = Sequential()

# TODO: Add GRU layers and Dropout layers to the model
# Note: Limit the total number of model parameters to 10,000
# Your code here:
model_gru.add(
    GRU(8, return_sequences=False, input_shape=(sequence_length, len(sequence_columns)))
)
# model_gru.add(Dropout(0.5))
# model_gru.add(GRU(16, return_sequences=False))
# model_gru.add(Dropout(0.5))


# Add a Dense output layer with sigmoid activation
model_gru.add(Dense(units=nb_out, activation="sigmoid"))

# Compile the model with binary crossentropy loss and Adam optimizer
model_gru.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# TODO: Print the model summary
print(model_gru.summary())

checkpoints_folder = config["checkpoints_folder"]
if not os.path.exists(checkpoints_folder):
    print(f"Creating {checkpoints_folder}")
    os.mkdir(checkpoints_folder)
gru_name = "gru_best.keras"
history = model_gru.fit(
    sequence_array,
    label_array,
    epochs=config["epochs"],  # TODO: Replace with the desired number of training epochs
    batch_size=config["batch_size"],  # TODO: Replace with the desired batch size
    validation_split=config[
        "validation_split"
    ],  # TODO: Replace with the desired validation split proportion
    verbose="auto",  # TODO: Replace with the desired verbosity level
    callbacks=[
        # TODO: Early stopping callback to stop training when validation loss stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.00001,  # TODO: Replace with the minimum change in validation loss to qualify as improvement
            patience=15,  # TODO: Replace with the number of epochs to wait before stopping training
            verbose=1,  # TODO: Replace with the desired verbosity level
            mode="min",
        ),
        # TODO: Model checkpoint callback to save the best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoints_folder, gru_name
            ),  # TODO: Replace with the file path to save the best model
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,  # TODO: Replace with the desired verbosity level
        ),
    ],
)

# TODO: summarize history for Accuracy
# TODO: Plot the training & validation accuracy over epochs and display the plot
# TODO: Save the plot to a file
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("GRU Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()


# TODO: summarize history for Loss
# TODO: Plot the training & validation loss over epochs and display the plot
# TODO: Save the plot to a file
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("GRU Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
fig_name = "GRU_Validation_vs_Training.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()


# TODO: Use the evaluate method to calculate the accuracy of the model on the training data
scores = model_gru.evaluate(
    sequence_array, label_array
)  # TODO: Replace with the correct code to evaluate the model on the training data

# Print the accuracy of the model on the training data
print(f"Training Accuracy: {scores[1]}")

# make predictions and compute confusion matrix
# TODO: Use the predict method to make predictions on the training data
# TODO: Convert the predicted probabilities to class labels (e.g., using a threshold of 0.5)
y_pred = model_gru.predict(
    sequence_array
)  # TODO: Use predict and convert probabilities to class labels
y_pred = (y_pred > 0.5).astype(int)
y_true = label_array

# TODO: Create a Pandas DataFrame from the predicted labels and save it to a CSV file
test_set = pd.DataFrame(
    y_pred, columns=["predicted_labels"]
)  # TODO: Replace with the correct code to create a DataFrame from the predicted labels
gru_predictions = "GRU"
gru_full_path = os.path.join(
    config["save_csv_folder"], gru_predictions, "gru_predicted_labels_train.csv"
)
test_set.to_csv(
    gru_full_path,
    index=False,
)

print("Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels")
# TODO: Compute the confusion matrix using confusion_matrix from sklearn.metrics
cm = confusion_matrix(
    y_true, y_pred
)  # TODO: Replace with the correct code to compute the confusion matrix
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="g", cmap="Blues", cbar=False
)  # fmt='g' to display integer values
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix of GRU")
fig_name = "GRU_Confusion_Matrix.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()
# TODO: Calculate the precision using precision_score and recall using recall_score from sklearn.metrics
precision = precision_score(
    y_true, y_pred
)  # TODO: Replace with the correct code to calculate precision
recall = recall_score(
    y_true, y_pred
)  # TODO: Replace with the correct code to calculate recall
print("precision = ", precision, "\n", "recall = ", recall)


# TODO: Pick the last sequence for each id in the test data
seq_array_test_last = []  # Replace with code to select last sequence for each id
for engine_id in test_df["id"].unique():
    sub_test_df = test_df[test_df["id"] == engine_id]

    last_sequence = sub_test_df[sequence_columns].values[-sequence_length:]
    if last_sequence.shape[0] >= sequence_length:
        seq_array_test_last.append(last_sequence)

# TODO: Convert to numpy array and ensure float32 data type
seq_array_test_last = np.array(seq_array_test_last).astype(np.float32)

# TODO: Pick the labels for the selected sequences
y_mask = (
    test_df.groupby("id").size() >= sequence_length
)  # TODO: Replace with code to select labels for sequences with length >= sequence_length
label_array_test_last = (
    []
)  # TODO: Replace with code to select labels for the selected sequences
for engine_id in test_df["id"].unique():
    if y_mask[engine_id]:  # Only consider engines with length >= sequence_length
        # Filter the data for the current engine_id
        sub_test_df = test_df[test_df["id"] == engine_id]

        # Select the label for the last sequence
        last_label = sub_test_df["label1"].values[-1]

        # Append the last label to the list
        label_array_test_last.append(last_label)

# Reshape and ensure float32 data type
label_array_test_last = np.array(label_array_test_last)
label_array_test_last = label_array_test_last.reshape(
    label_array_test_last.shape[0], 1
).astype(np.float32)


# TODO: Load the saved model if it exists
model_path = os.path.join(checkpoints_folder, gru_name)
if os.path.isfile(model_path):
    estimator = load_model(
        model_path
    )  # TODO: Replace with code to load the saved model

# TODO: Evaluate the model on the test data
scores_test_gru = estimator.evaluate(
    seq_array_test_last, label_array_test_last
)  # TODO: Replace with code to evaluate the model on the test data
print("Accuracy: {}".format(scores_test_gru[1]))

# TODO: Make predictions and compute confusion matrix
y_pred_test = estimator.predict(
    seq_array_test_last
)  # TODO: Replace with code to make predictions and convert to class labels
y_pred_test = (y_pred_test > 0.5).astype(int)
y_true_test = label_array_test_last

# TODO: Create pandas dataframe of y_pred_test and save predictions to CSV file
test_set = pd.DataFrame(y_pred, columns=["predicted_labels"])
gru_full_path_test = os.path.join(
    config["save_csv_folder"], gru_predictions, "predicted_labels_test_gru.csv"
)
test_set.to_csv(
    gru_full_path_test,
    index=False,
)

# TODO: Compute confusion matrix
print("Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels")
cm = confusion_matrix(
    y_true_test, y_pred_test
)  # TODO: Replace with the correct code to compute the confusion matrix
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="g", cmap="Blues", cbar=False
)  # fmt='g' to display integer values
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix of GRU (test)")
fig_name = "GRU_Confusion_Matrix_testing.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()

# TODO: Compute precision, recall, and F1-score
precision_test_gru = precision_score(
    y_true_test, y_pred_test
)  # TODO: Replace with code to compute precision
recall_test_gru = recall_score(
    y_true_test, y_pred_test
)  # TODO: Replace with code to compute recall
f1_test_gru = (
    2 * precision_test_gru * recall_test_gru / (precision_test_gru + recall_test_gru)
)  # TODO: Replace with code to compute F1-score
print(
    "Precision: ",
    precision_test_gru,
    "\n",
    "Recall: ",
    recall_test_gru,
    "\n",
    "F1-score:",
    f1_test_gru,
)

# TODO: Plot predicted and actual data for visual verification
plt.figure(figsize=(10, 6))

# Plot actual values (ground truth)
plt.plot(label_array_test_last, "bo", label="Actual Data", markersize=4, linestyle="")

# Plot predicted values
plt.plot(y_pred_test, "rx", label="Predicted Data", markersize=4, linestyle="")

plt.title("Predicted vs Actual Data")
plt.xlabel("Sample Index")
plt.ylabel("Binary Value")
plt.legend()
fig_name = "GRU_Predicted_vs_Actual_test.png"
plt.savefig(os.path.join(config["save_fig_folder"], fig_name))
plt.show()
