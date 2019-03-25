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

w = tf.Variable(initial_value=np.random.normal(loc=0., scale=1., size=(2, 1)))
b = tf.Variable(initial_value=np.random.normal((1,)))

@tf.function
def linearModel(x):
    return tf.matmul(x, w) + b

def loss(y_true, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=logits, name="cross_entropy_loss"
        ))

@tf.function
def train(x, y, x_test, y_test, learning_rate):
    with tf.GradientTape() as t:
        logits = linearModel(x)
        current_loss = loss(y, logits)
    dW, db = t.gradient(current_loss, [w, b])
    w.assign_sub(learning_rate * dW)
    b.assign_sub(learning_rate * db)
    return current_loss

epochs = range(100)
for epoch in epochs:
    current_loss, test_accuracy = train(
        x_train, y_train.reshape((1000, 1)).astype(dtype=np.float), 
        x_test, y_test.reshape((100, 1)).astype(dtype=np.float),
        learning_rate=1)
    print("Epoch %02d: loss %.8f, test %.6f" % (epoch, current_loss))

tf.print(w)
tf.print(b)
