from setuptools import find_packages, setup

setup(name="contest",
      version="0.2",
      description="Tools for Weights & Biases DAVIS Video Segmentation Contest",
      url="https://github.com/wandb/davis-contest/",
      packages=find_packages(),
      python_requires=">=3.6.9",
      install_requires=[
            "numpy",
            "pandas",
            "Pillow",
            "wandb"
      ],
      extras_require={
            "keras": ["scikit-image", "tensorflow"],
            "torch": ["ptflops", "pytorch_lightning", "scikit-image", "torchvision"]
      }
      )
