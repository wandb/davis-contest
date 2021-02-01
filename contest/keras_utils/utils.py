import numpy as np


def to_numpy_int_arrays(outputs, scale=255):
  outputs_np = outputs.numpy()
  outputs_np *= scale
  outputs_np = outputs_np.astype(np.uint8)
  channel_axis = -1
  outputs_np = np.squeeze(outputs_np, axis=channel_axis)

  return outputs_np
