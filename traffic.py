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

    images = []  
    labels = []  

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        # Prüfe, ob der Pfad existiert
        if not os.path.isdir(category_path):
            continue
        
        # Lade alle Bilder in dieser Kategorie
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            
            # Lade das Bild mit OpenCV
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Ändere die Bildgröße auf 30x30 Pixel
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            # Speichere Bild und Label
            images.append(img)
            labels.append(category)
    
    return images, labels

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def get_model():
    model = Sequential()

    # Statt `input_shape` direkt im ersten Layer → Verwende Input()
    model.add(Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Erste Convolutional Layer mit 32 Filtern, Kernelgröße 3x3, Aktivierungsfunktion ReLU
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Zweite Convolutional Layer mit 64 Filtern
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dritte Convolutional Layer mit 128 Filtern
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening Layer zum Umwandeln der 2D-Matrizen in Vektoren
    model.add(Flatten())

    # Voll verbundene Dense Layer mit 128 Neuronen
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout zur Vermeidung von Overfitting

    # Ausgabeschicht mit so vielen Neuronen wie Kategorien und Softmax-Aktivierung
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))

    # Kompilieren des Modells mit Adam-Optimizer und kategorischer Kreuzentropie als Verlustfunktion
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # **Speichert das Modell automatisch in der neuen Keras-Version**
    model.save("model.keras")

    return model
 


if __name__ == "__main__":
    main()

    

