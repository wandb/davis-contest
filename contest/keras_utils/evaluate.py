from pathlib import Path

import numpy as np
import pandas as pd

from . import utils
from ..utils import image

def run(model, evaluation_dataset, num_images, output_dir):
  """Runs keras model on the data in evaluation dataset
  and saves the output in output_dir so it can be packaged into
  a result Artifact. See ..evaluate.make_result_artifact function.
  """
  output_dir = Path(output_dir)
  output_paths = pd.DataFrame([np.nan] * num_images, columns=["output"])

  ii = 0

  for jj in range(len(evaluation_dataset)):

    outputs = model(evaluation_dataset[jj])
    outputs = utils.to_numpy_int_arrays(outputs)

    for output in outputs:
      path = Path(image.save_from_array(output, output_dir, ii))

      path_in_artifact = path.relative_to(Path(output_dir).parent)
        
      output_paths["output"].iloc[ii] = str(path_in_artifact)
      ii += 1

  return output_paths
