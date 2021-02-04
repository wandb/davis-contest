import os
import warnings

import numpy as np
import pandas as pd
import wandb

from .utils import image


def iou_from_output(prediction, annotation):
  """Calculates the intersection over union (IoU) metric
  for two mask arrays with entries in 0-255.

  Parameters:
    prediction: np.uint8 array
      Predicted mask for image as integer array, values from 0 to 255
    annotation: np.uint8 array
      Ground truth mask as integer array, values 0 and 255.

  Returns:
    iou_score: float
      Ratio of the intersection of provided masks
      to the union of the provided masks,
      unless both are empty, in which case -1.0.
  """
  pred_binary, annotate_binary = to_binary(prediction), to_binary(annotation)
  iou_score = binary_iou(pred_binary, annotate_binary)
  return iou_score


def binary_iou(pred_binary, annotate_binary):
    """Calculates the ratio of the pixel count in the intersection
    to the pixel count in the union of two integer arrays
    containing only 0s and 1s, pred_binary and annotate_binary.

  Parameters:
    prediction: np.float array
      Predicted mask for image as float array of 0s and 1s
    annotate: np.float array
      Ground truth mask as float array of 0s and 1s

    Returns:
      iou_score: float
        Ratio of the intersection of provided masks
        to the union of the provided masks,
        unless both are empty, in which case -1.0.
    """
    intersection = pred_binary * annotate_binary
    union = pred_binary + annotate_binary - (intersection)

    intersection_size = intersection.sum()
    union_size = union.sum()

    iou = -1.
    if union_size != 0:
        iou = np.divide(intersection_size, union_size)
    return iou


def to_binary(output):
  return np.round(output / 255.)


def run_evaluation(output_paths, annotation_paths, max_index=None):
  """Evaluates the perfomance of a model by comparing output to ground truth annotations
  based on paths to image files.

  Parameters:
    output_paths: pd.Series
      Contains strings with paths to model outputs as png files
    annotation_paths: pd.Series
      Contains strings with paths to ground truth annotations as png files,
      or nulls where output and annotation don't align
    max_index: int or None
      Maximum index to range over in paths.
      Used for debugging purposes.

  Returns:
    evaluation: list[wandb.Image, wandb.Image, float]
      The first Image is the model output, the second is the ground truth
    metrics: dict[string: numeric or wandb.Media]
      Metrics from evaluation to log to Weights & Biases
  """

  max_index = max_index or len(annotation_paths) - 1

  evaluation = []
  for ii in range(max_index + 1):
    output_path, annotation_path = output_paths.iloc[ii], annotation_paths.iloc[ii]

    if pd.isna(output_path):
      continue

    model_outputs = image.load_to_array(output_path)
    annotation = image.load_to_array(annotation_path)

    iou_score = iou_from_output(model_outputs, annotation)

    model_outputs_im = wandb.Image(model_outputs, mode="L", caption="model output")
    annotation_im = wandb.Image(annotation, mode="L", caption="target")

    evaluation.append([model_outputs_im, annotation_im, float(iou_score)])

  metrics = extract_metrics(evaluation)

  return evaluation, metrics


def extract_metrics(evaluation):
  mean_iou = np.mean([row[-1] for row in evaluation
                      if row[-1] > -1.])

  return {"segmentation_metric": mean_iou,
          "mean_iou": mean_iou}


def name_submission(result_name, suffix=""):
  name = "".join(result_name.split(":")[:-1])
  name += "-submission"
  if suffix != "":
    name += "-" + "suffix"
  return name


def build_table(evaluation):
  evaluation_table = wandb.Table(columns=["out", "target", "iou_score"])
  for row in evaluation:
    evaluation_table.add_data(*row)
  return evaluation_table


def make_result_artifact(output_paths, name, output_dir="outputs",
                         metadata=None, output_paths_path=None):
  """Given the pd.DataFrame output_paths, generated a wandb.Artifact with name name
  and adds the contents of the output_dir to that Artifact.
  The output_paths pd.DataFrame is saved to the Artifact with relative path "paths.json".

  Metadata should include computational complexity information (e.g. FLOPs, parameter count).
  """
  if metadata is None:
    metadata = {}

  check_result_artifact_metadata(metadata)

  if output_paths_path is None:
    output_paths_path = os.path.join("wandb", "paths.json")

  output_paths.to_json(output_paths_path)

  result_artifact = wandb.Artifact(name=name, type="result",
                                   metadata=metadata)
  result_artifact.add_dir(output_dir, "outputs")
  result_artifact.add_file(output_paths_path, "paths.json")

  return result_artifact


def check_result_artifact_metadata(metadata):
  required_keys = ["nlfops", "nparams"]
  if not all(key in metadata.keys() for key in required_keys):
    warnings.warn("Result artifact metadata does not contain all required keys:"
                  f" {required_keys}.\n"
                  "Submissions without this metadata may be rejected.")
