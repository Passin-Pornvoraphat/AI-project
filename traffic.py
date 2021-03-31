import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    try:
        labels = list()
        images = list()
        
        # list all item in directory
        for directory in os.listdir(data_dir):
            # filter non numerical directory
            if directory.isnumeric():
                # create a path, ex gtsrb\1 , gtsrb\2
                path = os.path.join(data_dir, directory)
                # list all file in that directory
                for file in os.listdir(path):
                    # read a file
                    image = cv2.imread(os.path.join(path, file))
                    # resize a picture file
                    image_resize = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    # make it to the list
                    images.append(image_resize)
                    labels.append(int(directory))
        
        return images, labels
    except:         
        raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    try:
        # setup model
        model = tf.keras.models.Sequential([
            
            # first convolutional layer (40 filters using a 3x3 kernel) and pooling layer (2x2 pool size)
            tf.keras.layers.Conv2D(40, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # second convolutional layer (80 filters using a 4x4 kernel) and pooling layer (3x3 pool size)
            tf.keras.layers.Conv2D(80, (4, 4), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            
            # flattening layer
            tf.keras.layers.Flatten(),
            
            # three hidden layers with dropout
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            
            # an output layer with 43 units
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ])
        
        # Train neural network
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    except:
        raise NotImplementedError


if __name__ == "__main__":
    main()
