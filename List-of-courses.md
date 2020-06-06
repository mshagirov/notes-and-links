# Courses, tutorials, books, and other useful resources
---
[[Main page](README.md)]
## Contents
- [Tissue Mechanics](#tissue-mechanics)
  - [Tissue Mechanics Courses and Books](#tissue-mechanics-courses-and-books)
  - [Tissue Mechanics Tools](#tissue-mechanics-tools)
- [ML Courses and books](#ml-courses-and-books)
- [AI, Probability, Complexity and Computation](#ai-probability-complexity-and-computation)
  - [AI](#ai)
  - [Probability and Mathematics](#probability-and-mathematics)
  - [Information](#information)
  - [Complexity (From Finance and Economics to Fractals and Artificial Life)](#complexity)
- [Useful Resources](#useful-resources)
  - [Pytorch](#pytorch)
  - [Python and Computer Vision](#python-and-cv)
  - [LaTeX and Vim](#latex-and-vim)
  - [Containers (e.g. Docker)](#containers-eg-Docker)
  - [Miscellaneous](#miscellaneous)
  - [Security](#security)
- [Datasets](#datasets)

---

## Tissue Mechanics
[[Contents](#contents)]
### Tissue Mechanics Courses and Books
[up](#tissue-mechanics)
* [ ] "Biological Physics of the Developing Embryo" by Forgacs _et al._
* [ ] "Mechanics of the Cell" by David Boal (_Chs 2,5,6,8,9,10,12_).
* [ ] "Intermolecular and Surface Forces" by Israelachvili (_Chs 9(partially), 11, partially 13 and 14, __17__(Adhesion and wetting, start with this, I guess), partially 20 and 21_).
* [ ] Python-based CFD course (basics) [[lecture notes with codes](http://ohllab.org/CFD_course/index.html)]
* [ ] Surface Evolver these examples:{[5pb](http://facstaff.susqu.edu/brakke/evolver/downloads/5pb.fe), [loops](http://facstaff.susqu.edu/brakke/evolver/downloads/loops.fe), and more [examples](http://facstaff.susqu.edu/brakke/evolver/examples/examples.htm)}, and [tutorial](http://facstaff.susqu.edu/brakke/evolver/html/tutorial.htm) (included in the manual, can learn a lot about surface physical models from SE's *tutorial and these examples*).
* [ ] "Physical Biology of the Cell" by Rob Phillips *et al.*.
* [ ] "Random Walks in  Biology" by H.C. Berg.
* [ ] <a name="fenics">FEniCS</a>: python library for solving PDEs with finite element methods. [_Introduction tutorial_](https://fenicsproject.org/pub/tutorial/sphinx1/index.html), and _book_.
* [ ] Differential Geometry: a beginner's course [[playlist](https://www.youtube.com/watch?v=_mvjOoTieTk&list=PLIljB45xT85DWUiFYYGqJVtfnkUFWkKtP)]
### Tissue Mechanics Tools
[up](#tissue-mechanics)
* `Tyssue` vertex model-based python package (API) [[repo](https://github.com/DamCB/tyssue)] [[docs](https://tyssue.readthedocs.io/en/latest/index.html)]
* Tissue Analyzer plugin for Fiji [[repo](https://github.com/mpicbg-scicomp/tissue_miner/blob/master/MovieProcessing.md#TissueAnalyzer)]
* FEniCS (for introduction tutorial and manual see [FEniCS entry above](#fenics))
* Finite Elements (PDE solver) Python implementation [[SfePy](http://sfepy.org/doc/development.html)] [[SfePy repo](https://github.com/sfepy/sfepy)]

## ML Courses and books
[[Contents](#contents)]
* [ ] __CS224n__: Natural Language Processing with Deep Learning
[[lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)]
[[website](http://web.stanford.edu/class/cs224n/)]
* [ ] __Fast.ai__'s Practical Deep Learning for Coders, v3 [[website](https://course.fast.ai)] [[fastbook](https://github.com/fastai/fastbook)]
* [ ] Deep learning (w/ Pytorch, NYU) [[repo](https://github.com/Atcold/pytorch-Deep-Learning)] [[website](https://atcold.github.io/pytorch-Deep-Learning/)]
* [ ] __CS294__: Deep Unsupervised Learning [[spring 2019](https://sites.google.com/view/berkeley-cs294-158-sp19/home)] [[spring 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/home)]
* [ ] __CS224__: Machine Learning with Graphs [[website](http://web.stanford.edu/class/cs224w/index.html)]
* [ ] Deep RL (UC Berkeley)
[[lectures](https://www.youtube.com/playlist?list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&app=desktop)]
[[website](http://rail.eecs.berkeley.edu/deeprlcourse/)]
* __Ref. materials__:
  - "Mathematics For Machine Learning" [[book repo](https://github.com/mml-book/mml-book.github.io)]
  - Probability Theory (For Scientists and Engineers) M. Betancourt [[post](https://betanalpha.github.io/assets/case_studies/probability_theory.html)]; A Conceptual Introduction to Hamiltonian Monte Carlo [[paper](https://arxiv.org/abs/1701.02434)]; [[case study](https://betanalpha.github.io/assets/case_studies/falling.html#1_experimental_design)]
  - For reviewing numerical linear algebra [[Online Book](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md)]
* [ ] NIPS2019 Bayesian Deep Learning [talk1](https://slideslive.com/38921874/bayesian-deep-learning-1) [talk2](https://slideslive.com/38921875/bayesian-deep-learning-2) [talk3](https://slideslive.com/38921876/bayesian-deep-learning-3) [talk4](https://slideslive.com/38921877/bayesian-deep-learning-4)
* [ ] Machine Learning (Tom Mitchell) [[website](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml)] [[lectures](https://www.youtube.com/playlist?list=PLAJ0alZrN8rD63LD0FkzKFiFgkOmEtltQ)]
* [ ] Interpretability and Explainability in Machine Learning
[[website](https://interpretable-ml-class.github.io)]
* [ ] Neural Density Estimation and Likelihood-free Inference (by G. Papamakarios, with tutorials) [[thesis](https://arxiv.org/abs/1910.13233)]
* ML
  - _also see_ [[information](#information)]
  - A Comprehensive Guide to Machine Learning (Berkeley University) (Nasiriany _et al._)
  - Elements of Statistical Learning (Hastie *et al.*)
  - Pattern Recognition and Machine Learning (Bishop)
  - Machine Learning A Probabilistic Perspective (Murphy) [[notebook repo](https://github.com/probml/pyprobml/)]
  - Learning from data (book or course)
  [[website](https://work.caltech.edu/lectures.html#lectures)]
  - Information Theory, Inference, and Learning Algorithms (MacKay)
  - Guided Tour of Machine Learning in Finance (set of courses)
  [[coursera](https://www.coursera.org/learn/guided-tour-machine-learning-finance/home/welcome)]
* [ ] Fundamentals of RL [[coursera](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning)]
* [ ] "How Decision Trees Work" [[post](https://brohrer.github.io/how_decision_trees_work.html)]
* [x] CS231n: CNNs for Visual Recognition [[lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)] [[website](http://cs231n.stanford.edu)]
* [ ] Ancient Secrets of Computer Vision [[website](https://pjreddie.com/courses/computer-vision/)]
* [ ] "Computer Vision: A Modern Approach by Forsyth & Ponce" and related courses
[[CS131](http://vision.stanford.edu/teaching/cs131_fall1920/syllabus.html)] and
[[CS231A](http://web.stanford.edu/class/cs231a/course_notes.html)]

## AI, Probability, Complexity and Computation
[[Contents](#contents)]
### AI
[[up](#ai-probability-complexity-and-computation)]
* [ ] AI course 
[[Edx](https://courses.edx.org/courses/BerkeleyX/CS188.1x-4/1T2015/course/)]
[[website](http://ai.berkeley.edu/lecture_videos.html)]
* [ ] Computer science theory toolkit[[playlist](https://www.youtube.com/playlist?list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&app=desktop)] (brief overview of CS theory)

### Probability and Mathematics
[[up](#ai-probability-complexity-and-computation)]
* Linear Algebra
[[ocw](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/)]
[[A 2020 Vision of Linear Algebra (for reviewing)](https://ocw.mit.edu/resources/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)]
* Matrix Methods in Data Analysis, Signal Processing, and Machine Learning [[ocw](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/index.htm)]
* Linear Algebra and Differential Equations [[Linear Diff Eqns, ocw](https://ocw.mit.edu/resources/res-18-009-learn-differential-equations-up-close-with-gilbert-strang-and-cleve-moler-fall-2015/index.htm)]
* Computational Science and Engineering [[Course 1, ocw](https://ocw.mit.edu/courses/mathematics/18-085-computational-science-and-engineering-i-fall-2008/index.htm)] and [[Course 2, ocw](https://ocw.mit.edu/courses/mathematics/18-086-mathematical-methods-for-engineers-ii-spring-2006/index.htm)]
* Multivariable Calculus [[ocw](https://ocw.mit.edu/courses/mathematics/18-02-multivariable-calculus-fall-2007/video-lectures/)]
* Computational Probability and Inference
[[Edx](https://courses.edx.org/courses/course-v1:MITx+6.008.1x+3T2016/wiki/MITx.6.008.1x.3T2016/)]
* Probabilistic Programming and Bayesian Methods for Hackers
[[website](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)]
[[pyro](http://pyro.ai/examples/dmm.html)]
  - The beginners guide to Hamiltonian Monte Carlo
  [[post](https://bayesianbrad.github.io/posts/2019_hmc.html)]
* Introduction to Time Series Analysis [[NIST Engin and Stats Handbook Ch6](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm)]
* The Fat Tails Statistical Project (N.N. Taleb) [[website](https://www.fooledbyrandomness.com/FatTails.html)]
### Information
[[up](#ai-probability-complexity-and-computation)]
* "Information Theory, Pattern Recognition, and Neural Networks" course
[[website](http://www.inference.org.uk/itprnn_lectures/)]
### Complexity
[[up](#ai-probability-complexity-and-computation)]

__Finance and Economics__
* Complexity Economics [W. Brian Arthur (SFI)](http://tuvalu.santafe.edu/~wbarthur/)
* "Fractals and Scaling in Finance" by Benoit B. Mandelbrot

__Artificial Life__
* Introduction to Artificial Life [(post)](https://thegradient.pub/an-introduction-to-artificial-life-for-people-who-like-ai/)



## Useful Resources
[[Contents](#contents)]
### Pytorch
[up](#useful-resources)
* Pytorch `torch.utils.tensorboard`
[[docs](https://pytorch.org/docs/stable/tensorboard.html)]
* Pytorch Lightning-- structured pytorch code [[post](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)]
* Deep Graph Library (for pytorch) [[website](https://www.dgl.ai)]
* Pytorch on TPUs [[colab notebooks](https://github.com/pytorch/xla)]
* Skorch--A scikit-learn compatible neural network library that wraps PyTorch 
[[docs](https://skorch.readthedocs.io/)] (i.e. pytorch with scikit-learn's high level functionality)
* Migrating code from PyTorch to fast.ai [[example](https://github.com/fastai/fastai2/blob/master/nbs/migrating.ipynb)]
* Botorch—a library for Bayesian Optimization built on PyTorch [[repo](https://github.com/pytorch/botorch)]
* Pyro-probabilistic programming library built on PyTorch [[pyro.ai](http://pyro.ai)] [[repo](https://github.com/pyro-ppl/numpyro)] or a numpy based NumPyro [[repo](https://github.com/pyro-ppl/numpyro)]
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
* Introducing Label Studio, a swiss army knife of data labeling [[post](https://towardsdatascience.com/introducing-label-studio-a-swiss-army-knife-of-data-labeling-140c1be92881)]
* Visdom (live data tool for creating, organizing, and sharing visualizations; supports pytorch, numpy) [[repo](https://github.com/facebookresearch/visdom)]
* `itkwidgets` for Jupyter (__2D,3D data visualisation__) [[repo with instructions](https://github.com/InsightSoftwareConsortium/itkwidgets)]
* `mpld3` brindges `matplotlib` with `d3js` [[mpld3](https://mpld3.github.io/index.html)] *interactive plots, plots to html, etc.*
* Altair (declarative statistical visualization library) [[website](https://altair-viz.github.io)]
* "Dive Into Python 3" [[html-book](https://diveintopython3.problemsolving.io/index.html)]
* napari : a tool for browsing, annotating, and analyzing large multi-dimensional images (python package) [[napari : repo](https://github.com/napari/napari)] [[napari : tutorials](http://napari.org)] [[ImagePy : repo](https://github.com/Image-Py/imagepy/)] __(check out the `napari`)__
* Matplotlib plot annotations [[guide](https://matplotlib.org/3.1.1/tutorials/text/annotations.html#plotting-guide-annotation)]; 3D arrows and 3D annotations [[gist](https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c)]
* Scipy [[tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html)]
* PyDy (Python multibody dynamics simulation toolbox) [[docs](http://www.pydy.org/documentation.html)]
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
  * BayesNet—TikZ library for drawing Bayesian networks, and graphical models [[repo](https://github.com/jluttine/tikz-bayesnet)]
  * Using _LaTeX_ with Vim [[tutorial](https://castel.dev/post/lecture-notes-1/)]
* Using Vim:
  * [x] Vim cheatsheet [[website](https://vim.rtorr.com)]
  * ctrlp.vim (Vim plugin) [[docs](http://ctrlpvim.github.io/ctrlp.vim/)]
  * browsing API code with Vim (using `ctags`) [[How to Browse Fastai Source Code Using Vim](https://medium.com/@mck.workman/how-to-browse-fastai-source-code-using-vim-87d9ef595ce1)]
### Containers (e.g. Docker)
[up](#useful-resources)
* Getting started with Docker containers [[link](https://docs.docker.com/get-started/)]
* Deploying JupyterHub with Kubernetes on OpenStack [[post](https://blog.jupyter.org/how-to-deploy-jupyterhub-with-kubernetes-on-openstack-f8f6120d4b1)]
### Miscellaneous
[up](#useful-resources)
* Regexper regexp "Railroad Diagrams" generator
[[website](https://regexper.com)] (can be used to interpret regexps)
### Security
[up](#useful-resources)
* References:
  * [[Hacker101 resources](https://www.hacker101.com/resources#2)] and [[Hacker101 videos](https://www.hacker101.com/videos)]
  * [[The Modern JavaScript Tutorial](https://javascript.info)]
  * [[Networking terminology](https://www.digitalocean.com/community/tutorials/an-introduction-to-networking-terminology-interfaces-and-protocols)]  and [[List of common port numbers](https://www.utilizewindows.com/list-of-common-network-port-numbers/)]

## Datasets
[[Contents](#contents)]
* [DeepCell](http://deepcell.org/data) cell, and nuclei segmentation and tracking dataset 
("Accurate cell tracking and lineage construction in live-cell imaging experiments with deep learning" Erick Moen _et al._ 2019) [[biorXiv](https://www.biorxiv.org/content/10.1101/803205v2)]
[[model repo](https://github.com/vanvalenlab/deepcell-tf)]
* [Google Dataset Search](https://toolbox.google.com/datasetsearch) [[more about it](https://ai.google/tools/#datasets)]
* [data-science-bowl-2018](https://www.kaggle.com/c/data-science-bowl-2018/data) (Nuclei segmentation)
* [Fluorescence Microscopy Denoising (FMD) dataset](https://github.com/bmmi/denoising-fluorescence)
* [FishExplorer](https://github.com/xiuyechen/fishexplorer) (Zebrafish functional imaging data)
