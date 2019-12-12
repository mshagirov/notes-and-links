# Courses, tutorials, books, and other useful resources
---
[[Main page](README.md)]
## Contents
- [Tissue Mechanics](#tissue-mechanics)
  - [Mechanics Courses](#mechanics-courses)
  - [Tissue Mechanics Tools](#tissue-mechanics-tools)
- [ML Courses and books](#ml-courses-and-books)
- [Useful Resources](#useful-resources)
  - [Pytorch](#pytorch)
  - [Python and Computer Vision](#python-and-cv)
  - [LaTeX and Vim](#latex-and-vim)
  - [Containers (e.g. Docker)](#containers-eg-Docker)
  - [Miscellaneous](#miscellaneous)
- [Datasets](#datasets)

---

## Tissue Mechanics
[[Contents](#contents)]
### Mechanics Courses
[up](#tissue-mechanics)
* [ ] Python-based CFD course (basics) [[lecture notes with codes](http://ohllab.org/CFD_course/index.html)]
### Tissue Mechanics Tools
[up](#tissue-mechanics)
* [ ] Finite Elements (PDE solver) Python implementation [[SfePy](http://sfepy.org/doc/development.html)] [[SfePy repo](https://github.com/sfepy/sfepy)]
* [ ] Tissue Analyzer plugin for Fiji [[repo](https://github.com/mpicbg-scicomp/tissue_miner/blob/master/MovieProcessing.md#TissueAnalyzer)]

## ML Courses and books
[[Contents](#contents)]
* [ ] (!) Linear Algebra
[[ocw](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/)]
* [ ] For reviewing numerical linear algebra
[[Online Book](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md)]
* [ ] (!) Computational Probability and Inference
[[Edx](https://courses.edx.org/courses/course-v1:MITx+6.008.1x+3T2016/wiki/MITx.6.008.1x.3T2016/)]
* [ ] Probabilistic Programming and Bayesian Methods for Hackers
[[website](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)]
[[pyro](http://pyro.ai/examples/dmm.html)]
  - The beginners guide to Hamiltonian Monte Carlo
  [[post](https://bayesianbrad.github.io/posts/2019_hmc.html)]
* [ ] __Fast.ai__'s Practical Deep Learning for Coders, v3 [[website](https://course.fast.ai)]
* [ ] Ancient Secrets of Computer Vision [[website](https://pjreddie.com/courses/computer-vision/)]
* [ ] "Computer Vision: A Modern Approach by Forsyth & Ponce" and related courses
[[CS131](http://vision.stanford.edu/teaching/cs131_fall1920/syllabus.html)] and
[[CS231A](http://web.stanford.edu/class/cs231a/course_notes.html)]
* [ ] Machine Learning (Tom Mitchell) [[website](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml)] [[lectures](https://www.youtube.com/playlist?list=PLAJ0alZrN8rD63LD0FkzKFiFgkOmEtltQ)]
* [ ] Interpretability and Explainability in Machine Learning
[[website](https://interpretable-ml-class.github.io)]
* [ ] Deep Unsupervised Learning [[cs294-158-sp19](https://sites.google.com/view/berkeley-cs294-158-sp19/home)]
* [ ] Neural Density Estimation and Likelihood-free Inference (by G. Papamakarios, with tutorials) [[thesis](https://arxiv.org/abs/1910.13233)]
* [ ] AI course 
[[Edx](https://courses.edx.org/courses/BerkeleyX/CS188.1x-4/1T2015/course/)]
[[website](http://ai.berkeley.edu/lecture_videos.html)]
* [ ] (ML)
  - A Comprehensive Guide to Machine Learning (Berkeley University) (Nasiriany _et al._)
  - Elements of Statistical Learning (Hastie *et al.*)
  - Pattern Recognition and Machine Learning (Bishop)
  - Machine Learning A Probabilistic Perspective (Murphy) [[notebook repo](https://github.com/probml/pyprobml/)]
  - Learning from data (book or course)
  [[website](https://work.caltech.edu/lectures.html#lectures)]
  - Information Theory, Inference, and Learning Algorithms (MacKay)
  - Guided Tour of Machine Learning in Finance (set of courses)
  [[coursera](https://www.coursera.org/learn/guided-tour-machine-learning-finance/home/welcome)]
* [ ] CS224n: Natural Language Processing with Deep Learning
[[lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)]
[[website](http://web.stanford.edu/class/cs224n/)]
* [ ] Deep RL (UC Berkeley)
[[lectures](https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&app=desktop)]
[[website](http://rail.eecs.berkeley.edu/deeprlcourse/)]
* [ ] "Information Theory, Pattern Recognition, and Neural Networks" course
[[website](http://www.inference.org.uk/itprnn_lectures/)]
* [ ] Fundamentals of RL [[coursera](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning)]
* [x] CS231n: CNNs for Visual Recognition [[lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)] [[website](http://cs231n.stanford.edu)]

## Useful Resources
[[Contents](#contents)]
### Pytorch
[up](#useful-resources)
* Pytorch `torch.utils.tensorboard`
[[docs](https://pytorch.org/docs/stable/tensorboard.html)]
* Pytorch on TPUs [[colab notebooks](https://github.com/pytorch/xla)]
* Skorch--A scikit-learn compatible neural network library that wraps PyTorch 
[[docs](https://skorch.readthedocs.io/)] (i.e. pytorch with scikit-learn's high level functionality)
* Botorch—a library for Bayesian Optimization built on PyTorch [[repo](https://github.com/pytorch/botorch)]
* Hydra--a framework for configuring complex applications [[link](https://cli.dev/docs/intro)], _e.g. use this to sweep parameters for models_.
* Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups
[[post](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)]
* `TensorFlow 2.0` + `Keras` Overview for Deep Learning Researchers
[[notebook](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO#scrollTo=zoDjozMFREDU)]
* Hangar (version control for data)
[[docs](https://hangar-py.readthedocs.io/en/latest/readme.html)]
* Programmatically Building and Managing Training Data
[[website](https://www.snorkel.org)]
### Python and CV
[up](#useful-resources)
* `itkwidgets` for Jupyter (__2D,3D data visualisation__) [[repo with instructions](https://github.com/InsightSoftwareConsortium/itkwidgets)]
* Altair (declarative statistical visualization library) [[website](https://altair-viz.github.io)]
* "Dive Into Python 3" [[html-book](https://diveintopython3.problemsolving.io/index.html)]
* napari : a tool for browsing, annotating, and analyzing large multi-dimensional images (python package) [[napari : repo](https://github.com/napari/napari)] [[napari : tutorials](http://napari.org)] [[ImagePy : repo](https://github.com/Image-Py/imagepy/)] __(check out the `napari`)__
* Matplotlib plot annotations [[guide](https://matplotlib.org/3.1.1/tutorials/text/annotations.html#plotting-guide-annotation)]
* Computer Vision, Deep Learning, and OpenCV on `PyImageSearch` [[tutorial](https://www.pyimagesearch.com/start-here/)] (_includes details on environment set up, hardware, and dataset preparation_)
* Jupyter notebook version control with jupytext, and automation with papermill [[post](https://medium.com/capital-fund-management/automated-reports-with-jupyter-notebooks-using-jupytext-and-papermill-619e60c37330)]
* IPython websites [[repo](https://github.com/stephenslab/ipynb-website)]
* Packaging and versioning (python related)
  * [x] Semantic Versioning [[post](https://semver.org)]
  * [ ] Pip distribution and packaging [[docs](https://packaging.python.org/guides/distributing-packages-using-setuptools/#choosing-a-versioning-scheme)]
  * `pip-tools`[[repo](https://github.com/jazzband/pip-tools)]
* Advanced numpy [[notebook](https://nbviewer.jupyter.org/github/vlad17/np-learn/blob/master/presentation.ipynb)]
* Guided filter (He et al.) python implement-n [[repo](https://github.com/swehrwein/python-guided-filter)]
### LaTeX and Vim
[up](#useful-resources)
* _LaTeX_ notes:
  * The Not So Short Introduction to  [[PDF](https://tobi.oetiker.ch/lshort/lshort.pdf)]
  * Beamer presentation (LaTeX) tutorial [[link](https://www.overleaf.com/learn/latex/Beamer_Presentations:_A_Tutorial_for_Beginners_(Part_1)—Getting_Started)]
  * Using _LaTeX_ with Vim [[tutorial](https://castel.dev/post/lecture-notes-1/)]
  * [x] Vim cheatsheet [[website](https://vim.rtorr.com)]
### Containers (e.g. Docker)
[up](#useful-resources)
* Getting started with Docker containers [[link](https://docs.docker.com/get-started/)]
* Deploying JupyterHub with Kubernetes on OpenStack [[post](https://blog.jupyter.org/how-to-deploy-jupyterhub-with-kubernetes-on-openstack-f8f6120d4b1)]
* Website security etc. tutorials and other readings [[hacker101](https://www.hacker101.com/resources)]
### Miscellaneous
[up](#useful-resources)
* The Modern JavaScript Tutorial [[website](https://javascript.info)]
* Regexper regexp "Railroad Diagrams" generator
[[website](https://regexper.com)] (can be used to interpret regexps)

## Datasets
[[Contents](#contents)]
* [DeepCell](http://deepcell.org/data) cell, and nuclei segmentation and tracking dataset 
("Accurate cell tracking and lineage construction in live-cell imaging experiments with deep learning" Erick Moen _et al._ 2019) [[biorXiv](https://www.biorxiv.org/content/10.1101/803205v2)]
[[model repo](https://github.com/vanvalenlab/deepcell-tf)]
* [Google Dataset Search](https://toolbox.google.com/datasetsearch) [[more about it](https://ai.google/tools/#datasets)]
* [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data) (Nuclei segmentation)
* [Fluorescence Microscopy Denoising (FMD) dataset](https://github.com/bmmi/denoising-fluorescence)
* [FishExplorer](https://github.com/xiuyechen/fishexplorer) (Zebrafish functional imaging data)
