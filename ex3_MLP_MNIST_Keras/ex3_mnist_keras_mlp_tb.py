# For auto-completion to work
import os
from datetime import datetime
from sklearn.metrics import classification_report
import tensorflow.python.keras as keras

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

    # Profile the training procedure, create tensorboard callback
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', 'mnist_keras_mlp', stamp)
    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logdir',
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq = 10000,
        profile_batch= 10000
    )

    model.fit(
        x=x_train, y=y_train, epochs=5, callbacks=[tb_callback]
    )
    y_pred = model.predict_classes(x=x_test)
    print(classification_report(y_test, y_pred, target_names=[
        '%d' % i for i in range(10)
    ]))
