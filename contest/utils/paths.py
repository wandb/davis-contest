import os

import pandas as pd


def artifact_paths(artifact, column=None):
  """From an artifact, get the paths of associated DAVIS files.
  As a side effect, downloads the Artifact to local file storage
  if it is not present.
  
  Parameters:
    artifact: wandb.Artifact
    column: None or string
      If string, name of column of paths to fetch from Artifact's paths_df
      If None, return the entire dataframe of paths
      
  Returns:
    paths: pd.DataFrame or pd.Series
      Paths to files attached to Artifact, sorted by index.
  """
  directory = artifact.download()
  paths = rebase_paths(get_paths(artifact, column), directory)
  return paths


def rebase_paths(paths, rebase_dir):
  """Takes a pd.Series or pd.DataFrame of paths
  and prepends the rebase_dir to the front of each.
  """
  if isinstance(paths, pd.Series):
    method = paths.map
  elif isinstance(paths, pd.DataFrame):
    method = paths.applymap

  return method(lambda s: os.path.join(rebase_dir, s))


def get_paths(artifact, column=None):
  """Returns, as a pd.Series or pd.DataFrame,
  the information about paths to associated DAVIS files
  for the provided wandb.Artifact, optionally from a specific column.
  
  The returned pandas object is always sorted by index.
  """
  try:
    paths_filename = artifact.get_path("paths.json").download()
  except KeyError:
    raise Exception("Artifact did not contain a path.json file at top-level.\n"
                    "All dataset and result Artifacts need a paths.json file listing associated files.")

  paths_df = pd.read_json(paths_filename)
  paths_df.sort_index(inplace=True)
  
  if column is not None:
    try:
      paths_series = paths_df[column]
    except KeyError:
      raise KeyError(f"could not find column {column} in {paths_filename}")
  else:
    return paths_df

  return paths_series
