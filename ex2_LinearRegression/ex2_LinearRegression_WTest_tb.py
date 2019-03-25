import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_dataset(mean_a=[0, 2], mean_b=[2, 0],
                   std_a=1., std_b=1., 
                   train_size=1000, test_size=100):
    train_samples = int(0.5 * train_size)
    test_samples = int(0.5 * test_size)
    train_size = train_samples * 2
    test_size = test_samples * 2
    total_samples = train_samples + test_samples
    cov_a = (std_a * std_a) * np.eye(2)
    cov_b = (std_b * std_b) * np.eye(2)
    a_samples = np.random.multivariate_normal(mean_a, cov_a, total_samples)
    b_samples = np.random.multivariate_normal(mean_b, cov_b, total_samples)
    train = np.block([
        [a_samples[:train_samples, :], np.zeros((train_samples, 1))],
        [b_samples[:train_samples, :], np.ones((train_samples, 1))],
    ])
    test = np.block([
        [a_samples[train_samples:, :], np.zeros((test_samples, 1))],
        [b_samples[train_samples:, :], np.ones((test_samples, 1))],
    ])
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train[:, 0:2], train[:, 2].astype(dtype=np.int)), \
           (test[:, 0:2], test[:, 2].astype(dtype=np.int))


def plot_dataset(x, y):
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr")
    plt.plot([-3, 6], [-3, 6], '--', c='k')
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)
    plt.show()

(x_train, y_train), (x_test, y_test) = create_dataset()
plot_dataset(x_train, y_train)

class LinearRegressionModel:
    def __init__(self):
        self.W = tf.Variable(
            initial_value=np.random.normal(loc=0., scale=1., size=(2, 1))
        )
        self.b = tf.Variable(initial_value=np.random.normal((1,)))
        self.trainable_variables = [self.W, self.b]

    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b

logdir = "../logs/ex2/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(
    os.path.abspath(os.path.join(logdir, "metrics"))
)
file_writer.set_as_default()

LRModel = LinearRegressionModel()
adagradOptimizer = tf.optimizers.Adagrad(learning_rate=0.7)

@tf.function
def loss(y_true, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=logits, name="cross_entropy_loss"
        ))

@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as t:
        y_logits = model(x)
        current_loss = loss(y, y_logits)
    gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss

@tf.function
def accuracy_metric(model, x, y):
    logits = model(x)
    y_pred = tf.round(tf.sigmoid(logits))
    correct_pred = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    return accuracy

@tf.function
def train(model, optimizer, x, y, x_test, y_test):
    current_loss = train_step(model, x, y, optimizer)
    train_accuracy = accuracy_metric(model, x, y)
    test_accuracy = accuracy_metric(model, x_test, y_test)
    return current_loss, train_accuracy, test_accuracy

epochs = range(100)
for epoch in epochs:
    if epoch == 0:
        tf.summary.trace_on(graph=True)
    current_loss, train_accuracy, test_accuracy = train(
        LRModel, adagradOptimizer,
        x_train, y_train.reshape((1000, 1)).astype(dtype=np.float), 
        x_test, y_test.reshape((100, 1)).astype(dtype=np.float))
    
    print("Epoch %02d: loss %.8f, train accuracy %.6f, test %.6f" % (
        epoch, current_loss, train_accuracy, test_accuracy
        ))
    if epoch == 0:
        tf.summary.trace_export(
            name="train",
            step=0,
            profiler_outdir=logdir)
    tf.summary.scalar('training accuracy', data=train_accuracy, step=epoch)
    tf.summary.scalar('test accuracy', data=test_accuracy, step=epoch)
    tf.summary.scalar('training loss', data=current_loss, step=epoch)
    
tf.print(w)
tf.print(b)
