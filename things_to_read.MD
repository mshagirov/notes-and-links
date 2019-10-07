# Papers, Blog Posts and Other Resources:

## Image Analysis and Segmentation
* [ ] Differentiable Mask-Matching Network (DMM-net)
[[repo](https://github.com/ZENGXH/DMM_Net)]
[[paper](https://www.cs.toronto.edu/~xiaohui/dmm/paper/dmmnet_iccv19.pdf)]
* [ ] Faster R-CNN
[[paper](https://arxiv.org/abs/1506.01497)]
[[pytorch](https://github.com/ZENGXH/faster-rcnn.pytorch)]
* [ ] Mask R-CNN
[[paper](https://arxiv.org/abs/1703.06870)]
[[repo](https://github.com/facebookresearch/maskrcnn-benchmark)]
* [ ] R-FCN (Region-based Fully Convolutional Networks)
[[paper](https://arxiv.org/abs/1605.06409)]
* [ ] Detectron (uses Mask R-CNN and others above)
[[repo](https://github.com/facebookresearch/Detectron)]
* [ ] Stardist
[[repo](https://github.com/mpicbg-csbd/stardist)]
* [ ] Focal Loss (addresses class imbalance of fg-bg)
[[paper](https://arxiv.org/abs/1708.02002)]
* [ ] Feature Pyramid Network (FPN)
[[paper](https://arxiv.org/abs/1612.03144)]
[[repo](https://github.com/jwyang/fpn.pytorch)]
* [ ] Segmentation-Enhanced CycleGAN
[[paper](https://www.biorxiv.org/content/10.1101/548081v1)] and read this
[[post](https://ai.googleblog.com/2019/08/an-interactive-automated-3d.html?m=1)]
[related: "High-precision automated reconstruction of neurons with flood-filling networks" (_Nature Methods_ __2018__)]
* [ ] Learning Fixed Points in Generative Adversarial Networks:
From Image-to-Image Translation to Disease Detection and Localization
[[paper](https://arxiv.org/abs/1908.06965)]
* [ ] Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data
[[paper](https://arxiv.org/abs/1909.09148)]
* [ ] Practical Full Resolution Learned Lossless Image Compression
[[repo](https://github.com/fab-jul/L3C-PyTorch#citation)]
* [ ] Learnable Triangulation of Human Pose
[[paper](https://arxiv.org/abs/1905.05754)]
* [ ] MONet: Unsupervised Scene Decomposition and Representation
[[paper](https://arxiv.org/abs/1901.11390)]

### Denoising and DIPs
* [ ] Deep Image Prior
[[website](https://dmitryulyanov.github.io/deep_image_prior)]
* [ ] Double DIP
[[website](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/)]
[[repo](https://github.com/yossigandelsman/DoubleDIP)] might need
[["Blind dehazing"](https://github.com/YuvalBahat/Dehazing-Airlight-estimation)]
* [ ] Noise2Self
[[paper](https://arxiv.org/abs/1901.11365)] and
[[repo](https://github.com/czbiohub/noise2self)]
* [ ] pN2V
[[PN2V repo](https://github.com/juglab/pn2v)] might help to revise
[[N2V repo](https://github.com/juglab/n2v)] and
[[N2N repo](https://github.com/NVlabs/noise2noise)]
[[paper](https://arxiv.org/abs/1803.04189)]

## Generative models
* [ ] GAN
[[paper](https://arxiv.org/abs/1406.2661)]
* [ ] Conditional GAN
[[paper](https://arxiv.org/abs/1411.1784?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=77587488&_hsenc=p2ANqtz--i5nQIm7lOwKMygW3rZvR9W1dgbq-yKtBIuLO0OdAbVFexTcWQvh6d5jHGk0Fj2Et8vhqTYcnuCs9ITplGKwlHIvmXag&_hsmi=77587488)]
* [ ] CycleGAN
[[website](https://junyanz.github.io/CycleGAN/)]
* [ ] StarGAN
[[paper](https://arxiv.org/abs/1711.09020)]
[[repo](https://github.com/yunjey/StarGAN)]
* [ ] PixelCNN++ 
[[paper](https://openreview.net/pdf?id=BJrFC6ceg)]
[[repo](https://github.com/openai/pixel-cnn)]
* [ ] PixelVAE++ 
[[paper](https://arxiv.org/abs/1908.09948)]
[[PixelVAE](https://arxiv.org/abs/1702.08658)]
[repo?]
* [ ] Progressive Growing of GANs for Improved Quality, Stability, and Variation
[[paper](https://arxiv.org/abs/1710.10196)]
* [ ] WaveNet
[[website](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)]
* "Generating Large Images from Latent Vectors" _see_ "Compositional Pattern Producing Networks" in "General Deep Learning" section.
* _See_ Attention (Transformers) in "General Deep Learning" section.

## Differentiable models
* [ ] Neural reparameterization
[[paper](https://arxiv.org/abs/1909.04240)]

## Other interesting papers
* [ ] Extracting 2D surface with PreMosa
[[PreMosa](https://cblasse.github.io/premosa/example.html)]
* [ ] Reinforcement Learning, Fast and Slow
[[paper](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30061-0)]

## General Deep Learning
* [ ] Highway nets
[[post](http://people.idsia.ch/~juergen/highway-networks.html)]
[[paper1](https://arxiv.org/abs/1507.06228)]
[[paper2](https://arxiv.org/abs/1612.07771)]
* [ ] An intriguing failing of convolutional neural networks and the CoordConv solution
* [ ] "A recipe for training neural networks" by A. Karpathy
[[post](http://karpathy.github.io/2019/04/25/recipe/)]
* "Compositional Pattern Producing Networks: A Novel Abstraction of Development"
  * [ ] [[paper](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)]
  * [ ] [[post1](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/),
  [post2](http://blog.otoro.net/2016/06/02/generating-large-images-from-latent-vectors-part-two/)]
* Attention ((Transformers):
  * [ ] Attention is all you need
  [[paper](https://arxiv.org/abs/1706.03762)]
  * [ ] Transformer-XL
  [[paper](https://arxiv.org/abs/1901.02860)]
* [ ] Image transformer (related to Transformer nets in NLP)
[[paper](https://arxiv.org/abs/1802.05751)]
* [ ] "The Building Blocks of Interpretability"
[[distill](https://distill.pub/2018/building-blocks/)]
* "The Unreasonable Effectiveness of Recurrent Neural Networks" by A. Karpathy
[[post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)]
* [ ] "Visualizing memorization in RNNs"
[[distill](https://distill.pub/2019/memorization-in-rnns/)]
* [ ] "Bias and Generalization in Deep Generative Models" by Zhao *et al.* 
[[post](https://ermongroup.github.io/blog/bias-and-generalization-dgm/)]
* [ ] "Exploring Neural Networks with Activation Atlases"
[[distill](https://distill.pub/2019/activation-atlas/)]
* [ ] "The Paths Perspective on Value Learning" (RL subproblem)
[[article](https://distill.pub/2019/paths-perspective-on-value-learning/)]
* [ ] "Human in the Loop: Deep Learning without Wasteful Labelling"
[[post](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html)]
* [ ] Domain-Adversarial Training of Neural Networks (Ganin Yaroslav, _et al._) 
[[paper1](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)]
* [ ] "Adversarial Examples Are Not Bugs, They Are Features"
[[post1](http://gradientscience.org/adv/)]
[[distill-discussion](https://distill.pub/2019/advex-bugs-discussion/)]
* [ ] "Domain Randomization for Sim2Real Transfer" by L. Weng
[[post](https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html)]
  * [ ] "Learning Dexterity"(OpenAI)
  [[post](https://openai.com/blog/learning-dexterity/)]
  * [ ] "MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies"
  [[website](https://xbpeng.github.io/projects/MCP/)]
* [ ] Adapting Pretrained Representations to Diverse Tasks
[[transfer learning](https://arxiv.org/pdf/1903.05987.pdf)]
* [ ] [World Models-- Can agents learn inside of their own dreams?](https://worldmodels.github.io)
* [ ] Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet
[[paper](https://arxiv.org/abs/1904.00760)]
* [ ] Visual exploration of gaussian processes
[[distill](https://distill.pub/2019/visual-exploration-gaussian-processes/)]
* [ ] Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks
[[paper](https://arxiv.org/abs/1905.11286)]
* [ ] Rectified Adam
[[post](https://medium.com/@lessw/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b)]
* [ ] Selective-Backprop (e.g. "Accelerating Deep Learning by Focusing on the Biggest Losers")
* [ ] Representer Point Selection for Explaining Deep Neural Networks
[[post](https://blog.ml.cmu.edu/2019/04/19/representer-point-selection-explain-dnn/)]
* [ ] Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly
[[paper](https://arxiv.org/abs/1903.06694)]
* [ ] Behaviour Suite for Reinforcement Learning
[[paper](https://arxiv.org/abs/1908.03568)]

## Useful Resources
* Pytorch `torch.utils.tensorboard`
[[docs](https://pytorch.org/docs/stable/tensorboard.html)]
* Hydra--a framework for configuring complex applications [[link](https://cli.dev/docs/intro)], _e.g. use this to sweep parameters for models_.
* `TensorFlow 2.0` + `Keras` Overview for Deep Learning Researchers
[[notebook](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO#scrollTo=zoDjozMFREDU)]
* Jupyter notebook version control with jupytext, and automation with papermill
[[post](https://medium.com/capital-fund-management/automated-reports-with-jupyter-notebooks-using-jupytext-and-papermill-619e60c37330)]
* IPython websites
[[repo](https://github.com/stephenslab/ipynb-website)]
* Hangar (version control for data)
[[docs](https://hangar-py.readthedocs.io/en/latest/readme.html)]
* Programmatically Building and Managing Training Data
[[website](https://www.snorkel.org)]
* Packaging and versioning (python related)
  * [x] Semantic Versioning [[post](https://semver.org)]
  * [ ] Pip distribution and packaging
  [[docs](https://packaging.python.org/guides/distributing-packages-using-setuptools/#choosing-a-versioning-scheme)]
  * `pip-tools`[[repo](https://github.com/jazzband/pip-tools)]
* Regexper regexp "Railroad Diagrams" generator
[[website](https://regexper.com)] (can be used to interpret regexps)
* _LaTeX_ notes:
  * The Not So Short Introduction to  [[PDF](https://tobi.oetiker.ch/lshort/lshort.pdf)]
  * Beamer presentation (LaTeX) tutorial
  [[link](https://www.overleaf.com/learn/latex/Beamer_Presentations:_A_Tutorial_for_Beginners_(Part_1)—Getting_Started)]
  * Using _LaTeX_ with Vim
  [[tutorial](https://castel.dev/post/lecture-notes-1/)] 
* Advanced numpy [[notebook](https://nbviewer.jupyter.org/github/vlad17/np-learn/blob/master/presentation.ipynb)]

