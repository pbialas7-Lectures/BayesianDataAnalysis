# Bayesian Data Analysis

Materials for my Bayesian Data Analysis course

## Setting up the python environment

In this course you will be working with python using jupyter notebooks or jupyter lab. So first you have to set up a proper python environment. I strongly encourage you to use some form of a virtual environment. You can use the [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or its smaller subset [miniconda](https://docs.conda.io/en/latest/miniconda.html), but personally I recommend using 
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge). 
After installing `mambaforge` create a new virtual environment `bda` (or any other name you want):

```
conda create -n bda python=3.12 ipython=8.33.0
```
Then activate the environment  by running
```
conda activate bda
```
Now you can install required packages (if you are using Anaconda some maybe already installed):

```
mamba install  jupyterlab jupytext myst-nb ipywidgets
mamba install numpy scipy  matplotlib
```
If you didn't install `mamba` then you can substitute `conda` for `mamba`. I tend to use `mamba` as it is markedly faster then `conda`.  
Finally run
```
pip install bda
```
to install the auxiliary package for this lecture.

After installing all required packages you can start `jupyter lab` by running 
```
jypyter lab
```

## Using python in lab

When using the computers in lab, please log to your linux account and then run
```
source /app/Python/3.10.4/VE/defaults/bin/activate
```
The you can run 
```
jupyter lab
```

## MyST format

The notebooks in the repository are stored in [MyST (Markedly Structured Text Format)](https://myst-parser.readthedocs.io/en/latest/) format. Thanks to the `jupytext` package you can open them right in the jupyter lab, by clicking the file name with righthand mouse button and choosing `open with` and then `Notebook`. If you are using jupyter notebook the you have to convert them prior to opening by running   
```shell
jupytext --to notebook <md file name>
```


## Using python in lab

To be described later

## Working with this repository

We will be using git and GitHub in this course. Please start by cloning this repository
```shell
git clone https://github.com/pbialas7-Lectures/BayesianDataAanalysis.git
```
Then change into just created repository and change the name of the remote
```shell
git remote rename origin lecture 
```
This will enable you to update your repository from mine by issuing command 
```shell
git pull lecture 
```
This will be needed, as I will be adding content to the repository throughout the semester. 
Please note that if you have changed some of the files e.g. some lectures then you  can have merge problems that you will have to resolve. So if you like to play with the notebooks is better to make a local copy and change that. 


Next please create a  **private** repository on [GitHub](https://github.com). Then add this repository as remote to the repository you have just cloned by running
```shell
git remote add origin  url_do_tego_repozytorium
```
You can find the url by clicking on the  `Code` button on the main page of the repository on GitHu. Then please push the content of the local repository using the command
```shell
git push -u origin main
```
Now you can use push your changes to repository  using `git push`. 

And finally please add me as a collaborator  (Select tab `Settings` and next `Collaborators and teams`). My  [GitHub id is `pbialas7`](https://github.com/pbialas7). 

## Jupytext and version control  (**!**)

While the notebook `*.ipynb` files are JSON text files, they are not well suited for version control. That is because each notebook stores additional  information, like e.g. the output of each cell and the number of times each cell was executed. As a consequence,  executing even a single cell changes the notebook without really changing it content and makes it very hard to keep it under version control. That's why it is recommended to use `jupytext` for this purpose. Jupytext automatically keeps the  `*.ipynb` notebook file in sync with another  text file that  does not contain this extra information. You can choose from several formats for this file. I am using the `*.md` files in [Myst Markdown](https://myst-parser.readthedocs.io/en/latest/) format. To use `jupytext` you have of course to install it (see above). After that please download the `jupytext.toml` file from my repository (you have it already if you have cloned my repository). Please download also the `.gitignore` file which will prevent the `*.ipynb` files to be recognised by `git`. Then  keep under version control **only** the `*.md` files. Do not add the `*.ipynb` files to your repository.  


