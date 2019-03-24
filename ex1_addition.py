import os
from datetime import datetime
import tensorflow as tf

"""
Build Graph
"""
a = tf.constant(value=[1., 2., 3., 4., 5.], name="var_a")
b = tf.Variable(initial_value=[1., 2., 3., 4., 5.], name="var_b")

@tf.function
def SimpleAdder(x):
    return a + b + x

"""
Visualization with TensorBoard
"""
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(os.path.abspath(logdir))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
print(SimpleAdder([-1., -2., -3., -4., -5.]))

with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=os.path.abspath(logdir))
