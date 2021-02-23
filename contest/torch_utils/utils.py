import os

import numpy as np
import torch
import wandb


def save_model_to_artifact(model, path, name, artifact_path="final_model"):
  """During a wandb.Run, save a model to path and as a wandb.Artifact
  and returns the resulting Artifact's complete identifier.

  See PyTorch documentation for details on saving and loading models:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
  """
  model_artifact = wandb.Artifact(name=name, type="trained-model")
  torch.save(model.state_dict(), path)
  model_artifact.add_file(path, artifact_path)
  wandb.run.log_artifact(model_artifact)

  return "/".join([wandb.run.entity, wandb.run.project, model_artifact.name])


def load_model_from_artifact(name, model_class, model_path="final_model", model_args=None, model_kwargs=None):
  """Pulls down the wandb.Artifact at name and loads the model state_dict at model_path,
  then feeds it to the provided model_class after passing in any model_args and model_kwargs.

  Model is placed in evaluation mode.

  See PyTorch documentation for details on saving and loading models:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
  """
  if model_args is None:
    model_args = []
  if model_kwargs is None:
    model_kwargs = {}

  model_artifact = wandb.run.use_artifact(name)
  model_artifact_dir = model_artifact.download()
  model_state_dict_path = os.path.join(model_artifact_dir, "final_model")

  model = model_class(*model_args, **model_kwargs)
  model.load_state_dict(torch.load(model_state_dict_path))
  model.eval()

  return model


def to_numpy_int_arrays(outputs, scale=255):
  """Convert PyTorch Module outputs from 0 to 1 float Tensors
  into np.uint8 arrays with values from 0 to scale (default 255).
  Removes (singleton) channel dimension (assumed shape BCHW, B optional).
  """
  outputs = outputs.detach().cpu()
  outputs_np = outputs.numpy()
  outputs_np *= scale
  outputs_np = outputs_np.astype(np.uint8)
  channel_axis = outputs_np.ndim - 3
  assert channel_axis >= 0, f"outputs must have ndim >=3 but had shape {outputs_np.shape}"
  if outputs_np.shape[channel_axis] != 1:
      raise ValueError("outputs should have a singleton channel axis "
                       f"but instead had size {outputs_np.shape[channel_axis]} "
                       "in axis {channel_axis}")
  outputs_np = np.squeeze(outputs_np, axis=channel_axis)

  return outputs_np
