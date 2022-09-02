# Introductory machine learning tutorials for accelerator physicists
## About
These tutorials were first used for the [2022 MT ARD ST3 pre-meeting Machine Learning Workshop](https://indico.desy.de/event/35272/)
, part of the [10th MT ARD ST3 Meeting 2022 in Berlin](https://indico.desy.de/event/33584/).

## Getting started

You can run the Notebooks in this repository either locally or in the cloud.

To run in the cloud, visit the [repository home page](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop) and click on the Binder badge. This may take a minute to load.

To run the notebooks for this workshop locally, please install *Python*. Then go ahead and clone the workshop repository and changing into the workshop directory.

```bash
git clone https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop.git
cd 2022-MT-ARD-ST3-ML-workshop
```

With Python installed, run the following command to install the packages for the workshop for the workshop.

```bash
pip install -r requirements.txt
```

Then start the Jupyter Notebook by running

```bash
jupyter notebook
```

You are now ready to execute the workshop notebooks. 🎉

**Note:** The reinforcement learning Notebook has a number of additional requirements and cannot be run on Binder. Please refer to the instructions in the Notebook itself for the installation. We do not require you to run this particular Notebook during the workshop.


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

### git not found

It may be that you cannot clone the repository to your local machine because git is not installed. In that case, instead of using git, simply visit the [repository home page](https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop), click the green *Code* button and select *Download ZIP*.

### Encountered error while trying to install package. box2d-py
You might encounter this error on a Linux distribution that does not come with *gcc* installed by default (such as Ubuntu). Run the following commands to fix, then re-run the command to create the environment.

```bash
sudo apt install build-essential
conda env remove --name mt-ard-st3-ml-workshop
```
