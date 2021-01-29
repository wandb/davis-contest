from pathlib import Path

import numpy as np
import PIL


def save_from_array(arr, folder, index):
  im = PIL.Image.fromarray(arr)
  path = folder / (str(index).zfill(5) + ".png")
  im.save(path)
  return str(path)

  
def load_to_array(path):
    im = PIL.Image.open(path)
    arr = np.array(im)
    return arr
