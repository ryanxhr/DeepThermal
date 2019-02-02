import tensorflow as tf
import numpy as np
import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import base as base_layer
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


class _LayerRNNCell(RNNCell):
  """Subclass of RNNCells that act like proper `tf.Layer` objects.

  For backwards compatibility purposes, most `RNNCell` instances allow their
  `call` methods to instantiate variables via `tf.get_variable`.  The underlying
  variable scope thus keeps track of any variables, and returning cached
  versions.  This is atypical of `tf.layer` objects, which separate this
  part of layer building into a `build` method that is only called once.

  Here we provide a subclass for `RNNCell` objects that act exactly as
  `Layer` objects do.  They must provide a `build` method and their
  `call` methods do not access Variables `tf.get_variable`.
  """

  def __call__(self, inputs, state, scope=None, *args, **kwargs):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size, self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size, s] for s in self.state_size`.
      scope: optional cell scope.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    # Bypass RNNCell's variable capturing semantics for LayerRNNCell.
    # Instead, it is up to subclasses to provide a proper build
    # method.  See the class docstring for more details.
    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)


class SimulatorRNNCell(_LayerRNNCell):
    """
    coaler RNN: (external_input_t, coaler_hidden_t-1 , coaler_action_t) --> (coaler_hidden_t, coaler_cell_t)
    burner RNN: (coaler_hidden_t, burner_hidden_t-1 , burner_action_t) --> (burner_hidden_t, burner_cell_t)
    steamer RNN: (burner_hidden_t, steamer_hidden_t-1 , steamer_action_t) --> (steamer_hidden_t, steamer_cell_t)

    loss: sum of three parts
    part1: coaler_hidden_t, coaler_state_t
    part2: burner_hidden_t, burner_state_t
    part3: steamer_hidden_t, steamer_state_t
    """
    def __init__(self, cell_config,
                 keep_prob,
                 forget_bias=1.0,
                 activation=None,
                 reuse=None,
                 name=None):
        """
        Args:
          cell_config: simulator config
          num_units: list, [coaler_num_units, burner_num_units, steamer_num_units]
        """
        super(SimulatorRNNCell, self).__init__(_reuse=reuse, name=name)
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._external_state_pos = cell_config.external_state_pos
        self._coaler_state_pos = cell_config.coaler_state_pos
        self._coaler_action_pos = cell_config.coaler_action_pos
        self._burner_state_pos = cell_config.burner_state_pos
        self._burner_action_pos = cell_config.burner_action_pos
        self._steamer_state_pos = cell_config.steamer_state_pos
        self._steamer_action_pos = cell_config.steamer_action_pos

        self._external_state_size = cell_config.external_state_size
        self._coaler_state_size = cell_config.coaler_state_size
        self._coaler_action_size = cell_config.coaler_action_size
        self._burner_state_size = cell_config.burner_state_size
        self._burner_action_size = cell_config.burner_action_size
        self._steamer_state_size = cell_config.steamer_state_size
        self._steamer_action_size = cell_config.steamer_action_size

        # num_units: list, [coaler_num_units, burner_num_units, steamer_num_units]
        _num_units = cell_config.num_units  # TODO
        self._coaler_num_units = _num_units[0]
        self._burner_num_units = _num_units[1]
        self._steamer_num_units = _num_units[2]
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._input_keep_prob = self._output_keep_prob = keep_prob

    @property
    def state_size(self):
        c_tuple = tuple((self._coaler_num_units, self._burner_num_units, self._steamer_num_units))
        h_tuple = tuple((self._coaler_num_units, self._burner_num_units, self._steamer_num_units))
        return LSTMStateTuple(c_tuple, h_tuple)

    @property
    def output_size(self):
        return tuple((self._coaler_num_units, self._burner_num_units, self._steamer_num_units))

    def get_coaler_inputs(self, inputs):
        # coaler inputs contains external_input, coaler_state and coaler_action
        # input: (batch_size, feature_nums)
        external_input = tf.slice(inputs, [0, self._external_state_pos],
                                  [-1, self._external_state_size])

        coaler_state = tf.slice(inputs, [0, self._coaler_state_pos],
                                [-1, self._coaler_state_size])
        coaler_action = tf.slice(inputs, [0, self._coaler_action_pos],
                                 [-1, self._coaler_action_size])
        return tf.concat([external_input, coaler_state, coaler_action], axis=1)

    def get_burner_inputs(self, inputs):
        # burner inputs contains burner_state and burner_action
        # input: (batch_size, feature_nums)
        burner_state = tf.slice(inputs, [0, self._burner_state_pos],
                                [-1, self._burner_state_size])
        burner_action = tf.slice(inputs, [0, self._burner_action_pos],
                                 [-1, self._burner_action_size])
        return tf.concat([burner_state, burner_action], axis=1)

    def get_steamer_inputs(self, inputs):
        # steamer inputs contains steamer_state and steamer_action
        # input: (batch_size, feature_nums)
        steamer_state = tf.slice(inputs, [0, self._steamer_state_pos],
                                 [-1, self._steamer_state_size])
        steamer_action = tf.slice(inputs, [0, self._steamer_action_pos],
                                  [-1, self._steamer_action_size])
        return tf.concat([steamer_state, steamer_action], axis=1)

    def build(self, inputs_shape):
        # coaler
        external_input_depth = self._external_state_size
        coaler_input_depth = self._coaler_state_size + self._coaler_action_size
        self._coaler_kernel = self.add_variable(
            "coaler_kernel",
            shape=[external_input_depth + coaler_input_depth + self._coaler_num_units, 4 * self._coaler_num_units],
            initializer=orthogonal_lstm_initializer())
        self._coaler_bias = self.add_variable(
            "coaler_bias",
            shape=[4 * self._coaler_num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        # burner
        burner_input_depth = self._burner_state_size + self._burner_action_size
        self._burner_kernel = self.add_variable(
            "burner_kernel",
            shape=[burner_input_depth + self._burner_num_units + self._coaler_num_units, 4 * self._burner_num_units],
            initializer=orthogonal_lstm_initializer())
        self._burner_bias = self.add_variable(
            "burner_bias",
            shape=[4 * self._burner_num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        # steamer
        steamer_input_depth = self._steamer_state_size + self._steamer_action_size
        self._steamer_kernel = self.add_variable(
            "steamer_kernel",
            shape=[steamer_input_depth + self._steamer_num_units + self._burner_num_units, 4 * self._steamer_num_units],
            initializer=orthogonal_lstm_initializer())
        self._steamer_bias = self.add_variable(
            "steamer_bias",
            shape=[4 * self._steamer_num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size, s]` for each s in `state_size`.
        """
        # Try to use the last cached zero_state. This is done to avoid recreating
        # zeros, especially when eager execution is enabled.
        state_size = self.state_size
        is_eager = context.in_eager_mode()
        if is_eager and hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and
                last_dtype == dtype and
                last_state_size == state_size):
                return last_output
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            output = _zero_state_tensors(state_size, batch_size, dtype)
        if is_eager:
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    def call(self, inputs, state):
        # inputs: (external_input, coaler_input, burner_input, steamer_input)
        # state: (c, h) is a 3-D tensor
        # c: (c_coaler, c_burner, c_steamer)
        # h: (h_coaler, h_burner, h_steamer)
        # self._state_is_tuple is True for simplicity
        def _should_dropout(p):
            return (not isinstance(p, float)) or p < 1

        # input dropout
        if _should_dropout(self._input_keep_prob):
            inputs = nn_ops.dropout(inputs, keep_prob=self._input_keep_prob)

        coaler_inputs = self.get_coaler_inputs(inputs)
        burner_inputs = self.get_burner_inputs(inputs)
        steamer_inputs = self.get_steamer_inputs(inputs)

        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        c, h = state
        coaler_h, burner_h, steamer_h = h
        coaler_c, burner_c, steamer_c = c

        # coal mill model
        with tf.variable_scope('coaler'):
            # inputs = self.batch_normalization(inputs, 'coal_mill_bn')
            coaler_gate_inputs = math_ops.matmul(
                array_ops.concat([coaler_inputs, coaler_h], 1), self._coaler_kernel)
            coaler_gate_inputs = nn_ops.bias_add(coaler_gate_inputs, self._coaler_bias)

            coaler_i, coaler_j, coaler_f, coaler_o = array_ops.split(
                value=coaler_gate_inputs, num_or_size_splits=4, axis=one)

            coaler_forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=coaler_f.dtype)
            # Note that using `add` and `multiply` instead of `+` and `*` gives a
            # performance improvement. So using those at the cost of readability.
            add = math_ops.add
            multiply = math_ops.multiply
            coaler_new_c = add(multiply(coaler_c, sigmoid(add(coaler_f, coaler_forget_bias_tensor))),
                               multiply(sigmoid(coaler_i), self._activation(coaler_j)))
            coaler_new_h = multiply(self._activation(coaler_new_c), sigmoid(coaler_o))

        with tf.variable_scope('burner'):
            # inputs = self.batch_normalization(inputs, 'coal_mill_bn')
            # only dropout coaler output
            if _should_dropout(self._output_keep_prob):
                coaler_h = nn_ops.dropout(coaler_h, keep_prob=self._output_keep_prob)

            burner_gate_inputs = math_ops.matmul(
                array_ops.concat([burner_inputs, burner_h, coaler_h], 1), self._burner_kernel)
            burner_gate_inputs = nn_ops.bias_add(burner_gate_inputs, self._burner_bias)

            burner_i, burner_j, burner_f, burner_o = array_ops.split(
                value=burner_gate_inputs, num_or_size_splits=4, axis=one)

            burner_forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=burner_f.dtype)
            # Note that using `add` and `multiply` instead of `+` and `*` gives a
            # performance improvement. So using those at the cost of readability.
            add = math_ops.add
            multiply = math_ops.multiply
            burner_new_c = add(multiply(burner_c, sigmoid(add(burner_f, burner_forget_bias_tensor))),
                               multiply(sigmoid(burner_i), self._activation(burner_j)))
            burner_new_h = multiply(self._activation(burner_new_c), sigmoid(burner_o))

        with tf.variable_scope('steamer'):
            # inputs = self.batch_normalization(inputs, 'coal_mill_bn')
            # only dropout burner output
            if _should_dropout(self._output_keep_prob):
                burner_h = nn_ops.dropout(burner_h, keep_prob=self._output_keep_prob)

            steamer_gate_inputs = math_ops.matmul(
                array_ops.concat([steamer_inputs, steamer_h, burner_h], 1), self._steamer_kernel)
            steamer_gate_inputs = nn_ops.bias_add(steamer_gate_inputs, self._steamer_bias)

            steamer_i, steamer_j, steamer_f, steamer_o = array_ops.split(
                value=steamer_gate_inputs, num_or_size_splits=4, axis=one)

            steamer_forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=steamer_f.dtype)
            # Note that using `add` and `multiply` instead of `+` and `*` gives a
            # performance improvement. So using those at the cost of readability.
            add = math_ops.add
            multiply = math_ops.multiply
            steamer_new_c = add(multiply(steamer_c, sigmoid(add(steamer_f, steamer_forget_bias_tensor))),
                                multiply(sigmoid(steamer_i), self._activation(steamer_j)))
            steamer_new_h = multiply(self._activation(steamer_new_c), sigmoid(steamer_o))

        new_c = tuple((coaler_new_c, burner_new_c, steamer_new_c))
        new_h = tuple((coaler_new_h, burner_new_h, steamer_new_h))
        # concat_h = array_ops.concat([coaler_new_h, burner_new_h, steamer_new_h], axis=1)
        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state


def orthogonal_lstm_initializer():
    def orthogonal(shape, dtype=tf.float32, partition_info=None):
        # taken from https://github.com/cooijmanstim/recurrent-batch-normalization
        # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
        """ benanne lasagne ortho init (faster than qr approach)"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return tf.constant(q[:shape[0], :shape[1]], dtype)
    return orthogonal


