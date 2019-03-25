# For auto-completion to work
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.python.keras as keras

def inputFlatten(x):
    return tf.reshape(x, (28*28))

def hiddenLayer(x, W, b):
    return tf.nn.relu(tf.matmul(W, x) + b, name="hidden")

def outputLayer(x, W, b):
    return tf.matmul(W, x) + b

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Normalize image pixel intensity
    x_train = x_train / 255
    x_test = x_test / 255
    # Build models with keras
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x=x_train, y=y_train, epochs=5
    )
    y_pred = model.predict_classes(x=x_test)
    print(classification_report(
        y_test, y_pred, target_names=['%d' % i for i in range(10)],
        digits=5
    ))
