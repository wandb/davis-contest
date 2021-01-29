"""Utilities for counting parameters and operations of a torch.Module.
"""
import ptflops
import torch


def count_params(model):
  return sum(p.numel() for p in model.parameters)
  

def count_flops(model, device):
  with device:
    try:
      macs, _ = ptflops.get_model_complexity_info(model, (3, 853, 480), as_strings=False)
    except ZeroDivisionError:
      raise ValueError("failed to count model FLOPs")
    return macs // 2
