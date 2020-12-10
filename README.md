![VAME](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/VAME_Logo-1.png)
![workflow](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/workflow.png)

# Release notes for this fork of VAME:
* 12-10-20 - Added TimeSeriesKMeans cluster method. Also added a plot_transitions function to segment_behavior.py, to create .svg file with transition matrix plot. This adds the dependencies 'tslearn' and 'seaborn', both of which can be installed from conda-forge.
* 12-8-20 - Added 'step_size' and 'gamma' parameters to config file. The 'step_size' parameter determines how many training epochs occur between each decrease in learning rate. The gamma parameter is the multiplicative reduction in learning rate after every step_size iterations. Step_size defaults to 100, and gamma defaults to 0.2. In the main VAME repository these are the values used in the model.
* 11-30-20 - Minor updates include changing legend location in evaluation result figures. Also sets the nan_policy for t-test in extractResults function to 'omit', so nan clusters will be ignored rather than the t-test result being nan. In a future update empty clusters will likely be set to 0.
* 11-24-20 - Added vame/custom/alignVideos.py to produce egocentric alignment CSV files in accordance with update to main VAME repository.
* 11-2-20 - Added vame/custom/helperFunctions.py - These functions help with pre-processing of DLC data, e.g. dropping bodyparts if you don't want to include all in model. Also includes functions for extracting and concatenating results, performing statistics on motif usage between groups, and writing human-readable result CSV files.

# VAME in a Nutshell
VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/) based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every time step. 

![behavior](https://github.com/LINCellularNeuroscience/VAME/blob/master/Images/behavior_structure_crop.gif)

The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## Installation
To get started we recommend using [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6 or higher. 
Here, you can create a [virtual enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to store all the dependencies necessary for VAME.

* Install the current stable Pytorch release using the OS-dependent instructions from the [Pytorch website](https://pytorch.org/get-started/locally/). Currently, VAME is tested on PyTorch 1.5.
* Go to the locally cloned VAME directory and run `python setup.py install` in order to install VAME in your active Python environment.

## Getting Started
First, you should make sure that you have a GPU powerful enough to train deep learning networks. In our paper, we were using a single Nvidia GTX 1080 Ti to train our network. A hardware guide can be found [here](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/). Once you have your hardware ready, try VAME following the [workflow guide](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow).

## News
* November 2020: We uploaded an egocentric alignment [script](https://github.com/LINCellularNeuroscience/VAME/blob/master/examples/align_demo.py) to allow more researcher to use VAME
* October 2020: We updated our manuscript on [Biorxiv](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2)
* May 2020: Our preprint "Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion" is out! [Read it on Biorxiv!](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v1)

### Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/AlexEMG/DeepLabCut/).
As such, the VAME project management codebase has been adapted from the DeepLabCut codebase.
The DeepLabCut 2.0 toolbox is © A. & M. Mathis Labs [www.deeplabcut.org](www.deeplabcut.org), released under LGPL v3.0.

### References
VAME preprint: [Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v1)

### License: GPLv3
See the [LICENSE file](../master/LICENSE) for the full statement.
