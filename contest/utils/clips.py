"""Utilities for working with DAVIS contest data
as clips of associated frames, rather than just single images.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def split_on_clips(paths_df, columns=None, split=0.8):
  """Splits a DataFrame of paths to images into two pieces,
  train and holdout, while respecting clip differences.
  See get_clips for information on how clip differences are defined.
  
  Parameters:
    paths_df: pd.DataFrame
      DataFrame whose columns are collections of paths to files.
    columns: None or iterable of strings
      If None, uses default of get_clips, otherwise checks for clips
      based on information in all of columns (see get_clip).
      Inferred clip identity must agree across columns for all rows.
    split: float
      Fraction of clips to put into train split. Non-integer totals are rounded down.
      
  Returns:
    train_split: pd.DataFrame
      DataFrame of paths for training set. Index is reset to integers.
    holdout_split: pd.DataFrame
      DataFrame of paths for holdout set. Index is reset to integers.
  """
  clip_ids = get_clips(paths_df, columns)
  clip_names = clip_ids.unique()
  clip_count = len(clip_names)

  train_clip_count = int(split * clip_count)

  train_clip_names = pd.Series(clip_names).sample(train_clip_count)
  train_mask = clip_ids.apply(lambda id: str(id) in train_clip_names.unique())
  holdout_mask = ~train_mask

  train_split = paths_df[train_mask].reset_index(drop=True)
  holdout_split = paths_df[holdout_mask].reset_index(drop=True)
  
  return train_split, holdout_split


def get_clips(paths_df, columns=None):
  """Applies get_clip to each column from columns
  that is in paths_df and returns a pd.Series of clip ids.
  """
  if columns is None:
    columns = ["raw", "annotation"]

  clip_serieses = []
  for column in columns:
    clip_series = paths_df[column].apply(get_clip)
    clip_serieses.append(clip_series) 

  clips = confirm_clips(clip_serieses)

  assert not any(clips.isna())

  return clips


def get_clip(path):
  """Infers clip name from a path to a frame of the clip.
  All frames are stored in folders named after the clip.
  """
  path = Path(path)
  clip_name = path.parent.name
  return str(clip_name)


def confirm_clips(clip_serieses):
  """Checks that the clip identities inferred
  from each column match. Returns a pd.Series of clip ids
  that has the agreed clip id wherever the columns match
  and a np.nan wherever they disagree.
  """
  length = len(clip_serieses[0])
  clips = pd.Series([np.nan] * length)

  for ii in range(length):
    clip = clip_serieses[0][ii]
    if all([clip_series[ii] == clip
                for clip_series in clip_serieses]):
      clips.iloc[ii] = clip
    else:
      clips.iloc[ii] = np.nan

  return clips
