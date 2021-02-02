from setuptools import find_packages, setup

setup(name="contest",
      version="0.2",
      description="Tools for Weights & Biases DAVIS Video Segmentation Contest",
      url="https://github.com/wandb/davis-contest/",
      packages=find_packages(),
      python_requires=">=3.6.9",
      install_requires=[
            "numpy>=1.19.5",
            "pandas>=1.1.5",
            "Pillow>=7.0.0",
            "wandb"
      ],
      extras_require={
            "keras": ["scikit-image>=0.16.2", "tensorflow>=2.4.1"],
            "torch": ["ptflops", "pytorch_lightning", "scikit-image>=0.16.2", "torchvision>=0.8.1"]
      }
      )
