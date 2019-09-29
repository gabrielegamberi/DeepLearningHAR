# HAR CNN training

from time import time
import numpy as np
import pandas as pd
from utils.utilities import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

class HARKeras:

    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.model = self.optimizer = self.history = None

        self.X_train, labels_train, list_ch_train = read_data(data_path=path_to_dataset, split="train") # Train
        self.X_test, labels_test, list_ch_test = read_data(data_path=path_to_dataset, split="test")     # Test

        assert list_ch_train == list_ch_test, "Mistmatch in channels!"

        # Normalize
        self.X_train, self.X_test = standardize(self.X_train, self.X_test)

        # Train/Validation Split (If you don't want to split data, comment the following two lines, so as the variable "y_vld")
        self.X_tr, self.X_vld, lab_tr, lab_vld = train_test_split(
            self.X_train, labels_train, stratify=labels_train, random_state=123, test_size=0.15)

        '''
        # --------------------------
        # When we have 200 samples (just like in our experiment) we might want to use all of them
        # So the TRAIN / VALIDATION split wouldn't be necessary
        self.X_tr = self.X_train
        lab_tr = labels_train
        # --------------------------
        '''
        # Hyperparameters
        self.batch_size = 450       # Batch size previously set to 600
        self.seq_len = 128          # Number of steps
        self.learning_rate = 0.0001
        self.epochs = 100           # Previously set to 250
        self.n_classes = 8          # Number of classes
        self.n_channels = 9         # Number of files

        # One-hot encoding:
        self.y_tr = one_hot(lab_tr, self.n_classes)
        self.y_test = one_hot(labels_test, self.n_classes)

        # If you don't use the split, make sure to comment this line
        self.y_vld = one_hot(lab_vld)


        # Gather the activities
        self.activities = []
        print("OPENING: ",path_to_dataset+"activity_labels.txt")
        with open(path_to_dataset+"activity_labels.txt") as file:
            for line in file:
                field = line.split(' ')
                field[1] = field[1][:-1]
                self.activities.append(field[1])

    def build_graph(self):
        # Declare model
        self.model = tf.keras.models.Sequential()

        # Layer 1 (Convolutional Layer + MaxPooling  <- both 1D)
        self.model.add(tf.keras.layers.Conv1D(input_shape=(128, 9), filters=18, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"))

        # Layer 2
        self.model.add(tf.keras.layers.Conv1D(filters=36, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"))

        # Layer 3
        self.model.add(tf.keras.layers.Conv1D(filters=72, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"))

        # Layer 4
        self.model.add(tf.keras.layers.Conv1D(filters=144, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same"))

        # Flatten and add dropout
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dropout(0.5))

        # Prediction
        self.model.add(tf.keras.layers.Dense(self.n_classes, activation=tf.nn.softmax))

        # Print Model Summary
        print(self.model.summary())

    def train_network(self):
        self.model.compile(optimizer='adam',            # Optimizer (better version of SGD)
                      loss='categorical_crossentropy',  # Type of loss function to calculate the error
                      metrics=['accuracy'])             # Metrics we want to track

        self.history = self.model.fit(self.X_tr, self.y_tr,
                                      epochs=self.epochs,
                                      validation_data=(self.X_vld, self.y_vld), # Useful to verify whether we are overfitting
                                      batch_size=self.batch_size)

    def save_model(self, model_name):
        keras_file = "./keras_models/"+model_name
        self.model.save(keras_file+".model")
        tf.keras.models.save_model(self.model, keras_file+".h5")
        converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file+".h5")
        tflite_model = converter.convert()
        open("./keras_models/model.tflite","wb").write(tflite_model)

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model("./keras_models/"+model_name+".model")

    def show_history(self):
        if self.history is not None:
            # Summarize history for accuracy
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'valditation'], loc='upper left')
            plt.show()

            # Summarize history for loss
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        else:
            print("--- No available history ---")

    # Useful to establish wether we are overfitting (Validation acc. must be < training Acc.  & Validation loss > Training loss)
    def evaluate_model(self):
        # Calculate the validation loss (Degree of error) -> I want to minimize it
        validation_loss, validation_acc = self.model.evaluate(self.X_vld, self.y_vld)
        # I don't want a TOO-MUCH-DELTA between validation_acc & the training_acc (otherwise I'm overfitting)

    # Useful to make a prediction (this is just a "testing" function)
    def make_prediction(self, subject=0):
        predictions = self.model.predict([self.X_test]) #output: probability distributions
        index_p = np.argmax(predictions[subject])
        print("subject-{} Predicted {} -> {}".format(subject, index_p+1, self.activities[index_p]))
        #test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)

    # ------- CONFUSION MATRIX SECTION (Data must be loaded differently) ----------
    # Utility function to print the confusion matrix
    def calc_confusion_matrix(self, Y_true, Y_pred):
        Y_true = pd.Series([self.activities[y] for y in np.argmax(Y_true, axis=1)])
        Y_pred = pd.Series([self.activities[y] for y in np.argmax(Y_pred, axis=1)])
        return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])

    def print_heat_map(self, confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
        """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
        Arguments
        ---------
        confusion_matrix: numpy.ndarray
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
            Similarly constructed ndarrays can also be used.
        class_names: list
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize: tuple
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: int
            Font size for axes labels. Defaults to 14.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting confusion matrix figure
        """

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize=figsize)

        import seaborn as sns
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig

    def plot_confusion_matrix(self):
        # Make predictions
        predict = self.model.predict([self.X_test])
        conf_matrix = self.calc_confusion_matrix(self.y_test, predict)

        # Show Heat-map
        self.print_heat_map(conf_matrix, self.activities, (self.n_classes, self.n_classes), fontsize=10)
        plt.title("Confusion Matrix", fontsize=18)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("Actual", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Evaluate our model
        self.model.evaluate(self.X_test, self.y_test)


if __name__ == "__main__":
    keras_core = HARKeras("./UCIHAR/")

    print("\n--- Building Graph ---")
    keras_core.build_graph()

    print("\n--- Training our network ---")
    keras_core.train_network()

    print("\n--- Saving our model ---")
    keras_core.save_model("activities")

    print("\n--- Loading Model ---")
    keras_core.load_model("activities")

    print("\n--- Evaluating our model on Validation Set ---")
    keras_core.evaluate_model()

    print("\n--- Predicting on Test Set ---")
    keras_core.make_prediction()

    print("\n--- Generate Confusion Matrix ---")
    keras_core.plot_confusion_matrix()

    print("\n--- Showing History ---")
    keras_core.show_history()