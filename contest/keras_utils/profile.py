import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


def count_params(model):
  return model.count_params()

def count_flops(model, dummy_inputs, batch_size=None):
  """
  Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
  Ignore operations used in only training mode such as Initialization.
  Use tf.profiler of tensorflow v1 api.
  """
  if not isinstance(model, (keras.Sequential, keras.Model)):
      raise ValueError(
          "model argument must be tf.keras.Model or tf.keras.Sequential instance"
      )

  if batch_size is None:
      batch_size = 1

  if not isinstance(dummy_inputs, list):
    dummy_inputs = [dummy_inputs]

  # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
  # FLOPS depends on batch size
  inputs = [
      tf.TensorSpec([batch_size] + list(inp.shape[1:]), inp.dtype) for inp in dummy_inputs
  ]
  real_model = tf.function(model).get_concrete_function(inputs)
  frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

  # Calculate FLOPS with tf.profiler
  run_meta = tf.compat.v1.RunMetadata()
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  flops = tf.compat.v1.profiler.profile(
    graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
  )
  return flops.total_float_ops
