from pathlib import Path

import numpy as np
import pandas as pd


def split_on_clips(files_df, columns=None, split=0.8):
  clip_ids = get_clips(files_df, columns)
  clip_names = clip_ids.unique()
  clip_count = len(clip_names)

  train_clip_count = int(split * clip_count)

  train_clip_names = pd.Series(clip_names).sample(train_clip_count)
  train_mask = clip_ids.apply(lambda id: str(id) in train_clip_names.unique())
  holdout_mask = ~train_mask

  train_split = files_df[train_mask].reset_index(drop=True)
  holdout_split = files_df[holdout_mask].reset_index(drop=True)
  
  return train_split, holdout_split


def get_clips(files_df, columns=None):
  if columns is None:
    columns = ["raw", "annotation"]

  clip_serieses = []
  for column in columns:
    clip_series = files_df[column].apply(get_clip)
    clip_serieses.append(clip_series) 

  clips = confirm_clips(clip_serieses)

  assert not any(clips.isna())

  return clips


def get_clip(path):
  path = Path(path)
  clip_name = path.parent.name
  return str(clip_name)


def confirm_clips(clip_serieses):
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
