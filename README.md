# Densely-Annotated Video Segmentation Contest

Welcome to the
Weights & Biases [video segmentation contest](http://wandb.me/davis-benchmark)!

Your goal is to train a neural network model
that can select the foreground object from a video clip
like the one below:

<img src="https://github.com/wandb/davis-contest/blob/main/segment.gif?raw=true" width="300" />

This task is known as "primary object segmentation"
or "video object segmentation" (VOS).

The quality of segmentations will be assessed using
the [Intersection over Union (IoU) metric](https://www.coursera.org/lecture/convolutional-neural-networks/intersection-over-union-p9gxz).

To mimic the
[constraints of designing for limited compute](https://arxiv.org/abs/1709.09503),
like mobile devices,
you're required to keep your network's parameter count
below 50 million.
Tools for profiling networks built in Keras and PyTorch are
included in the contest repository.

# Prizes

The prizes will be online retail gift certificates.
For winners inside the United States, this will be an Amazon gift card.

- First prize - $1000 gift certificate
- Second prize - $500 gift certificate
- Third prize - $250 gift certificate

# How to Participate

The contest is open to Qualcomm employees only.

This competition is split into two phases:
1. a _training phase_, where you can train on
a public training set and compare
your performance to other participants
on a public validation set, and
2. a _test phase_, where a test dataset without labels
will be provided and participants will
submit their solutions to be scored on a private leaderboard.

**Prizes will be awarded based on performance during the test phase only**.
Be careful not to over-engineer your model on the training and validation data!
In large public competitions and in industrial machine learning,
this kind of over-fitting dooms many promising projects.

The test phase will begin at midnight Pacific time on
March 29th, 2021.
See the Timeline section below.

#### During the training phase

- [Sign up](https://wandb.ai/login?signup=true) for W&B using your Qualcomm email. _Note_: The contest is open to Qualcomm employees only.
- Check out the Colab notebook for your preferred framework
([PyTorch/Lightning](http://wandb.me/davis-starter-pt),
[TensorFlow/Keras](davis-starter-keras)) for some starter code,
then build on it with your own custom data pipelines, training schemes, and model architectures.
You can develop in Colab or locally (see **Installing the `contest` Package** below).
- Once you're happy with your trained model, produce your formatted results,
as described in the **Formatting Your Results** section below.
- Evaluate those results using the [evaluation notebook](http://wandb.me/davis-submit).
See that notebook for details on how results will be scored.
- Submit your evaluation run to [the public leaderboard](http://wandb.me/davis-leaderboard).

Submissions are manually reviewed and will be approved within two business days.

Submitting evaluation runs is a great way to ensure your code runs
smoothly on data in the format used in the test phase,
that your results are properly formatted,
and that your submissions are valid,
so make sure to do so!

#### During the testing phase
- Download the video clips for the test data set (link information TBA).
- Run your trained model on that data, producing formatted results,
just like in the training phase (see **Formatting Your Results** below).
- Submit your results run to the private leaderboard (link information TBA).

# Getting Help

New to online contests with W&B, deep learning, or video segmentation?
No problem!
We have posted resources to help you understand the W&B Python library,
deep learning frameworks, suitable algorithms,
and some articles on neural networks below under the Resources section below.

Questions? Use the #qualcomm-competition
[slack channel](http://wandb.me/slack),
or email contest@wandb.com.

# Installing the `contest` Package

This section provides instructions
for installing the `contest` package from
the [GitHub repository](https://github.com/wandb/davis-contest)
for this competition.

There are three versions of the package:
one that only installs the core tools,
for formatting results and managing dataset paths,
and two versions that provide extra tools
for getting started in two popular deep learning frameworks.

Check out the starter notebooks
([PyTorch](http://wandb.me/davis-starter-pt),
[Keras](http://wandb.me/davis-starter-keras))
to see how the package is used.

### Installing the core tools

The package can be installed with
[`pip`](https://pip.pypa.io/en/stable/),
the standard package installer for Python:

```
pip install "git+https://github.com/wandb/davis-contest.git#egg=contest"
```

### Installing Keras and PyTorch/Lightning tools

To install the `contest.keras` or `contest.torch` framework subpackages,
provide the name of the `framework` at the end of the `pip install` command,
using the
[optional dependencies syntax](https://creatronix.de/pip-optional-dependencies/):

```
pip install "git+https://github.com/wandb/davis-contest.git#egg=contest[framework]"
```

where `framework` is one of `keras`, `torch`, or `keras,torch`.

# Formatting Your Results

> **See the starter notebooks
([PyTorch](http://wandb.me/davis-starter-pt),
[Keras](http://wandb.me/davis-starter-keras))
for more, including screenshots and code,
detailing the construction and formatting of the results.**

Results are to be submitted in the form of a Weights & Biases Artifact.
W&B's Artifacts system
([docs](http://docs.wandb.com/artifacts))
provides methods for storing, distributing,
and version-controlling datasets, models, and other large files.
Artifacts are also used to distribute the
training, validation, and test datasets for this contest.
See
[this video tutorial](http://wandb.me/artifacts-video)
and
[associated Colab notebook](http://wandb.me/artifacts-colab)
for more on how to use Artifacts.

We provide utility functions to produce 
a results artifact from a directory of model outputs in the repository
[here](https://github.com/wandb/davis-contest/blob/main/contest/evaluate.py#L132).

The best way to check that your results are being formatted correctly is to run the
[submission notebook](http://wandb.me/davis-submit),
look through the table that it uploads to Weights & Biases,
and submit the run for approval.

#### Format of the results artifact

A results artifact _must_ contain at least the following:
- a file called `paths.json`, containing a key `"output"` whose value is a dictionary ("object" in JSON lingo) with keys that are integer strings and values that are strings defining paths to files,
- at each path, a PNG file representing the model's outputs for the input frame from the dataset with the same integer index. This PNG file should be greyscale/luminance, with each byte representing an unsigned 8-bit integer (the `L` mode in [PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)), and
- in the [metadata](https://docs.wandb.ai/artifacts/api#2-create-an-artifact), the key `nparams`, counting the number of parameters in the model (including _all_ components).

The `paths.json` file can be generated easily
by saving a
[pandas `DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
with an integer index and a column called `"output"` with the
[`.to_json`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html#pandas.DataFrame.to_json)
method.
See the code in the starter notebooks and repository for examples,
including for how to create the W&B Artifact.

The contents of the PNG files will be used to
See [the evaluation code in the `contest` package](https://github.com/wandb/davis-contest/edit/main/contest/evaluate.py)
for details on how results will be scored,
in particular the functions
`iou_from_output` and `binary_iou`.

# Timeline

- Feburary 16 - Contest announced, training phase begins, public leaderboard opens
- March 29, 12:00am Pacific - training phase ends, test phase begins: test set made available for inference, private leaderboard opens
- March 31, 11:59pm Pacific - test phase ends: private leaderboard closes to new submissions
- Mid-April - Winners announced
- Early May - Retrospective webinar

# Other Rules

See
[Contest Terms & Conditions](http://wandb.me/davis-contest-terms)
for details,
including eligibility requirements and locations.

- You are free to use any framework you feel comfortable in, but you are responsible for accurately counting parameters.
- You may only submit results from one account.
- You can submit as many runs as you like.
- You can share small snippets of the code online or in our Slack community, but not the full solution. You should keep your Weights & Biases project set to [private](https://wandb.ai/site/data).
- You may similarly use snippets of code from online sources, but the majority of your code should be original. Originality of solution will be taken into account when scoring submissions. Submissions with insufficient novelty will be disqualified.

# Colab Notebooks
## Click the Badges Below to Access the Colab Notebooks

These Google Colab notebooks describe how to get started with the contest and submit results.

| Notebook    | Link |
|-------------|------|
| Get Started in PyTorch  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/davis-starter-pt) |
| Get Started in Keras  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/davis-starter-keras) |
| Evaluate Your Results  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/davis-submit) |
| Using Pretrained Networks  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/davis-pretrain) |

### Iterating quickly in Colab
Google Colab is a convenient hosted environment
you can use to run the baseline and iterate on your models quickly.

To get started:
- Open the baseline notebook you'd like to work with from the table above.
- Save a copy in Google Drive for yourself.
- To ensure the GPU is enabled, click Runtime > Change runtime type.
Check that the "hardware accelerator" is set to GPU.
- Step through each section, pressing play on the code blocks to run the cells.
- Add your own data engineering and model code.
- Review the Getting Started section for details on how to submit results to the public leaderboard.

# Ask Questions
If you have any questions, please feel free to email us at contest@wandb.com
or join our [Slack community](http://wandb.me/slack)
and post in the channel for this competition: `#qualcomm-competition`.

# Resources
- The [Weights & Biases docs](https://docs.wandb.com/library/python)
- The [paper describing the training and validation set](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)
- [PapersWithCode benchmark for training and validation set](https://paperswithcode.com/sota/video-object-segmentation-on-davis-2016)
- A [paper on 3DC-Seg, the current best-performing method](https://arxiv.org/abs/2008.11516) for this task. NOTE: this method uses almost triple this contest's parameter budget. You can use the approach for inspiration, but you'll need to look for ways to cut the parameter count.
- A [paper on MATNet, another state-of-the-art method](https://arxiv.org/abs/2003.04253) for this task. NOTE: as above, the parameter count is way above the limit -- 2x.
- A [paper on designing ConvNets for mobile devices](https://arxiv.org/abs/1709.09503)
