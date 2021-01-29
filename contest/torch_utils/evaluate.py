from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import utils
from ..utils import clips, image, paths


def run(model, dataloader, num_images, output_dir):
  """Runs torch.Module model on the data in DataLoader
  and saves the output in output_dir so it can be packaged into
  a result Artifact. See ..evaluate.make_result_artifact function.
  """
  output_dir = Path(output_dir)
  output_paths = pd.DataFrame([np.nan] * num_images, columns=["output"])
  
  with torch.no_grad():
    ii = 0
    for eval_batch in iter(dataloader):
    
      outputs = model.forward(eval_batch)
      outputs = utils.to_numpy_int_arrays(outputs)

      for output in outputs:
        path = Path(image.save_from_array(output, output_dir, ii))
        
        # ensure that path inside artifact is set correctly: drop everything before output_dir
        path_in_artifact = path.relative_to(Path(output_dir).parent)
        
        output_paths["output"].iloc[ii] = str(path_in_artifact)
        ii += 1
        
  return output_paths
