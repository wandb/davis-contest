"""Tools for working with data for the DAVIS contest
using the PyTorch and PyTorch Lightning libraries.
"""
import pytorch_lightning as pl
import skimage.io
import torch
from torchvision import transforms

from ..utils import clips


default_image_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
default_mask_transform = transforms.Compose([
      transforms.ToTensor(),
    ])


class VidSegDataset(torch.utils.data.Dataset):
  """From a pd.DataFrame of paths to image files and (optionally)
  to segmentation annotation images for those images,
  creates a simple subclass of torch.utils.data.Dataset suitable for use in
  a Video Segmentation task.
  """
  def __init__(self, paths_df, has_annotations=True, image_transform=None, mask_transform=None):
    self.paths_df = paths_df
    self.has_annotations = has_annotations

    self.image_paths = self.paths_df["raw"]
    if self.has_annotations:
      self.annotation_paths = self.paths_df["annotation"]

    if image_transform is None:
      self.image_transform = default_image_transform
    else:
      self.image_transform = image_transform

    self.mask_transform = mask_transform 

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.to_list()

    img_name = self.image_paths.iloc[idx]
    img = skimage.io.imread(img_name)

    if self.has_annotations:
      annotation_name = self.annotation_paths.iloc[idx]
      annotation = skimage.io.imread(annotation_name)

    if self.image_transform is not None:
      img = self.image_transform(img)
    if self.mask_transform is not None:
      annotation = self.mask_transform(annotation)

    if self.has_annotations:
      sample = img, annotation
    else:
      sample = img

    return sample
    
    
class VidSegDataModule(pl.LightningDataModule):
  """From a pd.DataFrame of paths to training images and their annotations,
  and optionally another pd.DataFrame of paths to holdout images and their annotations,
  generates a pl.LightningDataModule suitable for training a torch.Module on the
  training images and validating it on the holdout images.
  
  If only a single pd.DataFrame is provided, that pd.DataFrame is split into two,
  with the fraction put into the training split given by the split argument.
  
  See the PyTorch Lightning docs for details on pl.LightningDataModule:
    https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html?highlight=lightningdatamodule
    
  WARNING: this pl.LightningDataModule sets state in the setup method,
  and so is not suitable for multi-GPU training.
  """

  def __init__(self, training_paths_df,
               holdout_paths_df=None, split=0.8,
               num_workers=1, batch_size=None,
               image_transform=default_image_transform,
               mask_transform=default_mask_transform):
    super().__init__()

    if batch_size is None:
      self.batch_size = len(training_paths_df)
    else:
      self.batch_size = batch_size

    self.training_paths_df = training_paths_df
    if holdout_paths_df is not None:
      self.holdout_paths_df = holdout_paths_df
      self.is_split = True
    else:
      self.is_split = False
      self.split = 0.8

    self.num_workers = num_workers 

    self.image_transform = image_transform
    self.mask_transform = mask_transform

  def setup(self, stage=None):
    if not self.is_split:
      self.training_paths_df, self.holdout_paths_df = clips.split_on_clips(
        self.training_paths_df, split=self.split)
      self.is_split = True

    self.training_data = VidSegDataset(
        self.training_paths_df,
        image_transform=self.image_transform,
        mask_transform=self.mask_transform
    )

    self.holdout_data = VidSegDataset(
        self.holdout_paths_df,
        image_transform=self.image_transform,
        mask_transform=self.mask_transform
    )

  def prepare_data(self, stage=None):
    pass

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.holdout_data, batch_size=self.batch_size,
                                       num_workers=self.num_workers)
