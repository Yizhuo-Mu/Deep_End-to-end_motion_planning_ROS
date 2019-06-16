from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 77500
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20000


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.

def _generate_laser_and_cmd_batch(laser, goal,cmd, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    laser, goal, cmd = tf.train.shuffle_batch(
        [laser,goal, cmd],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    laser, goal, cmd = tf.train.batch(
        [laser, goal,cmd],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
  return laser, goal,cmd


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(laser,goal):
  """Build the ResNet model.
  Args:
    laser: LiDAR
    goal: target relative position
  Returns:
    .
  """
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7,1,64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv1d(laser, kernel, 3, padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm1 = tf.layers.batch_normalization(pre_activation)
    conv1 = tf.nn.relu(norm1, name=scope.name)
    _activation_summary(conv1)

  # norm1
   # norm1 = tf.layers.batch_normalization(conv1,axis=0)

  # pool1
  pool1 = tf.layers.max_pooling1d(conv1, 3, strides=2,
                                  padding='same', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[3, 64, 64],
                                           stddev=5e-2,
                                           wd=None)
      conv = tf.nn.conv1d(pool1, kernel, 1, padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      norm2 = tf.layers.batch_normalization(pre_activation)
      conv2 = tf.nn.relu(norm2, name=scope.name)
      _activation_summary(conv2)

  # norm2
     # norm2 = tf.layers.batch_normalization(conv2, axis=0)

  # conv3/block1
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,64,64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv1d(conv2, kernel, 1, padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm3 = tf.layers.batch_normalization(pre_activation)
    block1=pool1+norm3
    conv3 = tf.nn.relu(block1, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,64,64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv1d(conv3, kernel, 1, padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm4 = tf.layers.batch_normalization(pre_activation)
    conv4 = tf.nn.relu(norm4, name=scope.name)
    _activation_summary(conv4)

  # norm4
   # norm4 = tf.layers.batch_normalization(conv4, axis=0)

  #block2
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,64,64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv1d(conv4, kernel, 1, padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm5 = tf.layers.batch_normalization(pre_activation)
    block2=norm3+norm5
    conv5 = tf.nn.relu(block2, name=scope.name)
    _activation_summary(conv5)

  # pool2
  pool2 = tf.layers.average_pooling1d(conv5,3,2,padding='same',name='pool2')

  # fc1
  with tf.variable_scope('fc1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape_pool2 = tf.reshape(pool2, [FLAGS.batch_size, -1])
    reshape=tf.concat([reshape_pool2,goal],1)
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc1)

  # fc2
  with tf.variable_scope('fc2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024,1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
    _activation_summary(fc2)

  # fc3
  with tf.variable_scope('fc3') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                              stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    fc3 = tf.nn.relu(tf.matmul(fc2, weights) + biases, name=scope.name)
    _activation_summary(fc3)


  with tf.variable_scope('commandlayer') as scope:
    weights = _variable_with_weight_decay('weights', [512, 2],
                                          stddev=1/512.0, wd=None)
    biases = _variable_on_cpu('biases', [2],
                              tf.constant_initializer(0.0))
    commandlayer = tf.add(tf.matmul(fc3, weights), biases, name=scope.name)
    _activation_summary(commandlayer)

  return commandlayer


def loss(logits, cmd):

  cmd = tf.cast(cmd, tf.float32)
  loss_function = tf.abs(cmd-logits,name='loss_per_example')*10.0#/FLAGS.batch_size
  loss_mean = tf.reduce_mean(loss_function, name='loss')
  tf.add_to_collection('losses', loss_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train ResNet model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.

  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op