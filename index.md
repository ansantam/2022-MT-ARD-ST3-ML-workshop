# Introductory machine learning tutorials for accelerator physicists
## About
These tutorials were first used for the [2022 MT ARD ST3 pre-meeting Machine Learning Workshop](https://indico.desy.de/event/35272/)
, part of the [10th MT ARD ST3 Meeting 2022 in Berlin](https://indico.desy.de/event/33584/).

## Getting started
As a prerequisite for this workshop, please install *Anaconda*. You can refer to the [Anaconda website](https://www.anaconda.com/) for instructions.

Start by cloning the workshop repository and changing into the workshop directory.

```bash
git clone https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop.git
cd 2022-MT-ARD-ST3-ML-workshop
```

With Anaconda installed, run the following command in the workshop directory to create the environment for the workshop.

```bash
conda env create -f environment.yml
```

Use the following command to activate the workshop environment.

```bash
conda activate mt-ard-st3-ml-workshop
```

Then start the Jupyter Notebook by running

```bash
jupyter notebook
```

You are now ready to execute the workshop notebooks. 🎉


## Slides
 - [Introduction to machine learning in accelerator physics](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/slides/0-welcome.pdf)
 - [Introduction to artificial neural networks](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/slides/1-neural-networks.pdf)
 - [Introduction to Bayesian optimization](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/slides/2-bayesian-optimization.pdf)
 - [Application of Bayesian optimization to improve the injection efficiency at KARA ](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/slides/3-bo-kara-demo.pdf)

## Hands-on tutorials
- [Neural networks](https://nbviewer.org/github/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/1-neural_networks.ipynb)
- [Bayesian optimization](https://nbviewer.org/github/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/2-bayesian_optimization.ipynb)
- [Reinforcement learning](https://nbviewer.org/github/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/3-reinforcement_learning.ipynb)

### Bonus material
- [Basic reinforcement learning introduction without ML libraries (dynamic programming)](https://nbviewer.org/github/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/bonus_material/RL_simple_gridworld.ipynb)

## References
- [Literature: ML in accelerator physics](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop/blob/main/references/references.pdf)

## Troubleshooting

### Encountered error while trying to install package. box2d-py
You might encounter this error on a Linux distribution that does not come with *gcc* installed by default (such as Ubunut). Run the following commands to fix, then re-run the command to create the environment.

```bash
sudo apt install build-essential
conda env remove --name mt-ard-st3-ml-workshop
```
