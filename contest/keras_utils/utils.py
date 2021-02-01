import os

import numpy as np
import tensorflow.keras as keras
import wandb


def save_model_to_artifact(model_path, name, artifact_path="final_model"):
  """During a wandb.Run, save the model at model_path as a wandb.Artifact
  and returns the resulting Artifact's complete identifier.
  """
  model_artifact = wandb.Artifact(name=name, type="trained-model")
  model_artifact.add_file(model_path, artifact_path)
  wandb.run.log_artifact(model_artifact)
  
  return os.path.join(wandb.run.entity, wandb.run.project, model_artifact.name)
  
  
def load_model_from_artifact(name, model_path="final_model"):
  model_artifact = wandb.run.use_artifact(name)
  model_artifact_dir = model_artifact.download()

  model = keras.models.load_model(os.path.join(model_artifact_dir, model_path))
  return model


def to_numpy_int_arrays(outputs, scale=255):
  outputs_np = outputs.numpy()
  outputs_np *= scale
  outputs_np = outputs_np.astype(np.uint8)
  channel_axis = -1
  outputs_np = np.squeeze(outputs_np, axis=channel_axis)

  return outputs_np
