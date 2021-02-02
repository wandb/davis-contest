"""Tools for working with data for the DAVIS contest
using the Keras library.
"""
import math

import tensorflow.keras as keras
import numpy as np
import skimage.io

class VidSegDatasetSequence(keras.utils.Sequence):
  """From a pd.Series of paths to image files and (optionally)
  another pd.Series of segmentation annotation images for those images,
  creates a simple subclass of torch.utils.data.Dataset suitable for use in
  a Video Segmentation task.
  """

  def __init__(self, image_paths, annotation_paths=None, batch_size=32):
    self.image_paths, self.annotation_paths = image_paths, annotation_paths
    self.batch_size = batch_size

  def __len__(self):
    return math.ceil(len(self.image_paths) / self.batch_size)

  def __getitem__(self, idx):
    image_paths = self.image_paths.iloc[self._batch_start(idx): self._batch_start(idx + 1)]
    if self.annotation_paths is not None:
      annotation_paths = self.annotation_paths.iloc[self._batch_start(idx): self._batch_start(idx + 1)]
    else:
      annotation_paths = None

    images = np.array([skimage.io.imread(path) for path in image_paths])
    
    if annotation_paths is not None:
      annotations = [skimage.io.imread(path) for path in annotation_paths]
      annotations = np.array(annotations) / 255.
      return images, annotations

    else:
      return images

  def _batch_start(self, idx):
    return idx * self.batch_size
