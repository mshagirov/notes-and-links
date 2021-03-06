# Papers, Blog Posts and Other Resources:

\[[Main page](./README.md)\]

## Contents

1. [Image Analysis and Segmentation](#image-analysis-and-segmentation)
   1. [Deep Image Priors \(DIPs\)](#deep-image-priors)
2. [Graph Neural Networks](#graph-neural-networks)
3. [Generative models](#generative-models)
4. [Bayesian Approach to Deep Learning](#bayesian-approach-to-deep-learning)
5. [Sim2Real and transfer learning](#sim2real-and-transfer-learning)
6. [Recurrent Neural Networks](#recurrent-neural-networks)
7. [Attention, Transformers](#attention-transformers)
8. [General Deep Learning, and RL](#general-deep-learning-ml-and-rl)
9. [Force transduction, mechanobiology, and simulating physical systems](#force-transduction-mechanobiology-and-simulating-physical-systems)

## Image Analysis and Segmentation

\[[Contents](things_to_read.md#contents)\]

* [ ] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
* [ ] iGPT: [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) 
* [ ] [pix2vertex: Unrestricted Facial Geometry Reconstruction Using Image-to-Image Translation](https://arxiv.org/abs/1703.10131)
\[[pytorch repo](https://github.com/eladrich/pix2vertex.pytorch)\]
* [ ] Panoptic segmentation \[[paper](https://arxiv.org/abs/1801.00868)\] \(_see detectron2 for implementation_\)
* [ ] Faster R-CNN \[[paper](https://arxiv.org/abs/1506.01497)\] \[[pytorch](https://github.com/ZENGXH/faster-rcnn.pytorch)\]
* [ ] Focal Loss \(addresses class imbalance of fg-bg\) \[[paper](https://arxiv.org/abs/1708.02002)\]
* [ ] Learnable Triangulation of Human Pose [[website](https://saic-violet.github.io/learnable-triangulation)] [[paper](https://arxiv.org/abs/1905.05754)] [[repo](https://github.com/karfly/learnable-triangulation-pytorch/tree/master/mvn)]
* [x] \[[Noise2Self](https://arxiv.org/abs/1901.11365)\] \[[PN2V](https://github.com/juglab/pn2v)\] \[[N2V](https://arxiv.org/abs/1803.04189)\]
* [ ] On Network Design Spaces for Visual Recognition \[[paper](https://arxiv.org/abs/1905.13214)\] \[[repo](https://github.com/facebookresearch/pycls)\]

### Deep Image Priors

\[[Contents](things_to_read.md#contents)\]

* [x] Deep Image Prior \[[website](https://dmitryulyanov.github.io/deep_image_prior)\]
* [x] Double DIP \[[website](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/)\] \[[repo](https://github.com/yossigandelsman/DoubleDIP)\] might need \[["Blind dehazing"](https://github.com/YuvalBahat/Dehazing-Airlight-estimation)\]
* [x] Computational Mirrors: Blind Inverse Light Transport by Deep Matrix Factorization \[[paper](https://arxiv.org/abs/1912.02314)\]

## Graph Neural Networks

\[[Contents](things_to_read.md#contents)\]

* [x] Learning to Simulate Complex Physics with Graph Networks [[paper](https://arxiv.org/abs/2002.09405)]
* [x] Graph Convolutional Networks \(T. Kipf\) \[[post](http://tkipf.github.io/graph-convolutional-networks/)\]\[[paper](https://arxiv.org/abs/1609.02907)\]\[[repo\(pytorch\)](https://github.com/tkipf/pygcn)\]
* [x] Neural Relational Inference for Interacting Systems [[paper](https://arxiv.org/abs/1802.04687)]
* [x] [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://astroautomata.com//paper/symbolic-neural-nets/) [paper](https://arxiv.org/abs/2006.11287)
* [ ] Relational inductive biases, deep learning, and graph networks [[paper](https://arxiv.org/abs/1806.01261)]
* [ ] Understanding Attention and Generalization in Graph Neural Networks \[[paper](https://arxiv.org/abs/1905.02850)\]
* [x] Neural Message Passing for Quantum Chemistry \[[paper](https://arxiv.org/abs/1704.01212)\]
* [ ] Molecule Attention Transformer [[paper](https://arxiv.org/abs/2002.08264)]
* [ ] Deep Graph Infomax \[[paper](https://arxiv.org/abs/1809.10341)\]
* [ ] A Comprehensive Survey on Graph Neural Networks \[[paper](https://arxiv.org/abs/1901.00596)\]
* **Expressive power of GNNs**:
  * [ ] Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction [[paper](https://arxiv.org/abs/1802.05451)]
  * [ ] Deep Sets [[paper](https://arxiv.org/abs/1703.06114)]
* **Tutorials** :
  * [ ] Representation Learning on Networks \(e.g. GraphSAGE\) \[[slides](http://snap.stanford.edu/proj/embeddings-www/)\] \[[video from a related talk](https://www.youtube.com/watch?v=YrhBZUtgG4E)\]
  * [ ] \(by B. Knyazev\) \[[GNN introduction](https://medium.com/@BorisAKnyazev/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-1-3d9fada3b80d), [pytorch implementation examples](https://github.com/bknyaz/examples/blob/master/fc_vs_graph_train.py)\] \[[spectral GCN tutorial](https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801), [spectral GCN application paper](https://arxiv.org/abs/1811.09595)\] \[Image Classification with Hierarchical Multigraph Networks [\(paper w/ blog and repo\)](https://github.com/bknyaz/bmvc_2019)\]


## Generative models

\[[Contents](things_to_read.md#contents)\]

* [ ] Continual Unsupervised Representation Learning \[[paper](https://arxiv.org/abs/1910.14481)\]\[[repo](https://github.com/deepmind/deepmind-research/tree/master/curl)\]
* [ ] Learning Implicit Generative Models by Matching Perceptual Features \[[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/dos_Santos_Learning_Implicit_Generative_Models_by_Matching_Perceptual_Features_ICCV_2019_paper.html)\]
* [ ] GAN \[[paper](https://arxiv.org/abs/1406.2661)\]
* [ ] GAN hacks \[[post](https://github.com/soumith/ganhacks)\]
* [ ] Seeing What a GAN Cannot Generate \[[paper](https://arxiv.org/pdf/1910.11626.pdf)\]
* [ ] SinGAN: Learning a Generative Model from a Single Natural Image \[[paper](https://arxiv.org/abs/1905.01164)\]
* [ ] On Adversarial Mixup Resynthesis \[[video presentation](https://www.youtube.com/watch?v=ezbC3_VZeNY)\] \[[paper](https://arxiv.org/abs/1903.02709)\]
* [ ] Conditional GAN \[[paper](https://arxiv.org/abs/1411.1784?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=77587488&_hsenc=p2ANqtz--i5nQIm7lOwKMygW3rZvR9W1dgbq-yKtBIuLO0OdAbVFexTcWQvh6d5jHGk0Fj2Et8vhqTYcnuCs9ITplGKwlHIvmXag&_hsmi=77587488)\]
* [ ] CycleGAN \[[website](https://junyanz.github.io/CycleGAN/)\]
* [ ] StarGAN \[[paper](https://arxiv.org/abs/1711.09020)\] \[[repo](https://github.com/yunjey/StarGAN)\]
* [ ] PixelCNN++ \[[paper](https://openreview.net/pdf?id=BJrFC6ceg)\] \[[repo](https://github.com/openai/pixel-cnn)\]
* [ ] PixelVAE++ \[[paper](https://arxiv.org/abs/1908.09948)\] \[[PixelVAE](https://arxiv.org/abs/1702.08658)\] \[repo?\]
* [ ] Progressive Growing of GANs for Improved Quality, Stability, and Variation \[[paper](https://arxiv.org/abs/1710.10196)\]
* [ ] Few-shot Video-to-Video Synthesis \[[website with links](https://nvlabs.github.io/few-shot-vid2vid/)\]
* [ ] WaveNet \[[website](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)\]
* "Generating Large Images from Latent Vectors" _see_ "Compositional Pattern Producing Networks" in "General Deep Learning" section.
* _also see_ ["Attention, Transformers"](things_to_read.md#attention-transformers), and ["RNN"](things_to_read.md#recurrent-neural-networks)sections.

## Bayesian Approach to Deep Learning

\[[Contents](things_to_read.md#contents)\]

_Theory and review_:

* [ ] A Practical Bayesian Framework for Backpropagation Networks \(uses Laplace approximation\) \[[paper](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1992.4.3.448)\]
* [ ] Bayesian Learning via Stochastic Dynamics \(uses Hamiltonian Monte Carlo sampling\) \[[paper](https://papers.nips.cc/paper/613-bayesian-learning-via-stochastic-dynamics)\]
* [ ] Another complementary approach to the two above is the "Variational Bayes" approach.
* [ ] Uncertainty in Deep Learning \(PhD Thesis, 2016\) \[[blog w/ Thesis PDF links](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)\] \(**reviews all of the above**\)
* [ ] See abstracts and references in [http://bayesiandeeplearning.org](http://bayesiandeeplearning.org) \(2016-19 workshops\)

_Implementations_:

* [ ] Bayesian Learning via Stochastic Gradient Langevin Dynamics \[[paper](http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf)\] \(**very easy to implement with traditional SGD**\)
* [ ] Pyro \(and NumPyro\), or Botorch.
* [ ] Bayesian Active Learning \(Baal\) \(from ElementAI\) \[[repo with links](https://github.com/ElementAI/baal)\]—\[[docs with a list of bayesian active learning papers](https://baal.readthedocs.io/en/latest/)\].

## Sim2Real and transfer learning

\[[Contents](things_to_read.md#contents)\]

* [ ] Neocortical plasticity \(about unsupervised learning in neocortex\)\[[paper](https://openreview.net/pdf?id=S1g_N7FIUS)\]
* [ ] DiffTaichi: Differentiable Programming for Physical Simulation \[[paper](https://arxiv.org/abs/1910.00935)\] \[[colab](https://colab.research.google.com/github/znah/notebooks/blob/master/taichi_colab.ipynb)\]
* [ ] Domain-Adversarial Training of Neural Networks \(Ganin Yaroslav, _et al._\) \[[paper1](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)\]
* [x] "Domain Randomization for Sim2Real Transfer" by L. Weng \[[post](https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html)\]
* [ ] "Meta-Learning: Learning to Learn Fast" by L. Weng \[[post](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#define-the-meta-learning-problem)\]
* [ ] "Learning Dexterity"\(OpenAI\) \[[post](https://openai.com/blog/learning-dexterity/)\]
* [ ] Adapting Pretrained Representations to Diverse Tasks \[[transfer learning](https://arxiv.org/pdf/1903.05987.pdf)\]
* [ ] [World Models-- Can agents learn inside of their own dreams?](https://worldmodels.github.io)
* [ ] Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks \[[paper](https://arxiv.org/abs/1910.01663)\]
* [ ] "MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies" \[[website](https://xbpeng.github.io/projects/MCP/)\]
* [ ] Self-supervised representation learning \(by L. Weng\) \[[post](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)\]
* [ ] \(Model discovery\) Data-driven discovery of coordinates and governing equations \[[paper](https://www.pnas.org/content/116/45/22445)\]

## Recurrent Neural Networks

\[[Contents](things_to_read.md#contents)\]

* [ ] Attention and Augmented Recurrent Neural Networks \[[paper](https://distill.pub/2016/augmented-rnns/)\]
* [ ] Lilian Weng's blog post about attention[[post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#whats-wrong-with-seq2seq-model)]
* [ ] Generating Sequences With Recurrent Neural Networks A. Graves \[[paper](https://arxiv.org/abs/1308.0850)\] \(used in "Four Experiments in Handwriting with a Neural Network" \[[paper](https://distill.pub/2016/handwriting/)\]\) w/ implementation explained in a blog post \[[post](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/)\]
* [ ] Understanding LSTM Networks by C. Olah \[[post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)\]
* [ ] Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies \[[paper](https://arxiv.org/abs/1611.01368)\] \(_measuring compositional learning_\)
* [ ] "The Unreasonable Effectiveness of Recurrent Neural Networks" by A. Karpathy \[[post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)\]
* [ ] "Inferring single-trial neural population dynamics using sequential auto-encoders" \[[paper](https://www.nature.com/articles/s41592-018-0109-9)\] <a name="rnn-dynamical-system-nature">(RNNs to approximate neuron firing described as nonlinear dynamical systems)</a>
* [x] "Visualizing memorization in RNNs" \[[distill](https://distill.pub/2019/memorization-in-rnns/)\]

## Attention, Transformers

\[[Contents](things_to_read.md#contents)\]

* [ ] Attention is all you need \[[paper](https://arxiv.org/abs/1706.03762)\] \[[__annotated transformer__](http://nlp.seas.harvard.edu/2018/04/03/attention.html)\]
* [ ] Transformers: State-of-the-art Natural Language Processing \[[review](https://arxiv.org/abs/1910.03771)\]
* [ ] Tool for analysing transformers \[[exBERT](http://exbert.net/)\]
* [ ] Reformer \[[post](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html)\]
* [ ] Transformer-XL \[[paper](https://arxiv.org/abs/1901.02860)\]
* [ ] Write With Transformer \(GTP2-XL\) \[[website](https://transformer.huggingface.co)\] 
* [ ] NLP papers from hugging face \[[website](https://huggingface.co)\]
* [ ] Image transformer \(related to Transformer nets in NLP\) \[[paper](https://arxiv.org/abs/1802.05751)\]
* [ ] Stabilizing Transformers for Reinforcement Learning \[[paper](https://arxiv.org/abs/1910.06764)\]
* Attention and Augmented Recurrent Neural Networks \(go to [RNN section](things_to_read.md#recurrent-neural-networks)\)
* Lilian Weng's blog post about attention \(go to [RNN section](things_to_read.md#recurrent-neural-networks)\)

## General Deep Learning, ML and RL

\[[Contents](things_to_read.md#contents)\]

* General
  * [x] "A recipe for training neural networks" by A. Karpathy \[[post](http://karpathy.github.io/2019/04/25/recipe/)\]
  * [x] An intriguing failing of convolutional neural networks and the CoordConv solution
  * [x] [On the distance between two neural networks and the stability of learning](https://arxiv.org/abs/2002.03432)
      * deep NNs ( f(x|W) ) do not have Lipschitz continuous gradients in practice (Lipschitz: df/dW magnitude is limited) , which is assumed when using quadratic penalty on the weights (W^2 regul-n, "quadratic trust region")
      * for deep NNs "trust region" is quasi-exponential and depends on relative strengths of perturbations to each layer (e.g. in SGD it's df/dW).
      * the [proposed optimiser](https://github.com/jxbz/fromage) accounts for layer-wise learning rate differences, and uses only one hyper-param (lr); results in more stable learning when tested on multiple NN architectures/models.
  * [ ] [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) (_see also training in DIP paper_)
  * [ ] On the adequacy of untuned warmup for adaptive optimization \[[paper](https://arxiv.org/abs/1910.04209)\]
  * [ ] "The Building Blocks of Interpretability" \[[distill](https://distill.pub/2018/building-blocks/)\]
  * [ ] "Exploring Neural Networks with Activation Atlases" \[[distill](https://distill.pub/2019/activation-atlas/)\]
  * [ ] Distilling the Knowledge in a Neural Network \[[paper](https://arxiv.org/abs/1503.02531)\]
  * [ ] Weight Uncertainty in Neural Networks \[[paper](https://arxiv.org/abs/1505.05424)\]

\(_not sorted or read yet_\)

* [ ] Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One \[[paper](https://arxiv.org/abs/1912.03263)\]
* [ ] Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks \[[paper](https://arxiv.org/abs/1911.09737)\]
* [ ] Self-training with Noisy Student improves ImageNet classification \[[paper](https://arxiv.org/abs/1911.04252)\]
* [ ] Lottery Ticket Hypothesis [blog post with references](https://ai.facebook.com/blog/understanding-the-generalization-of-lottery-tickets-in-neural-networks/)
* [ ] Meta-Learning Deep Energy-Based Memory Models \[[paper](https://arxiv.org/abs/1910.02720)\]
* [ ] Why re-sampling imbalanced data isn’t always the best idea \[[post](https://stroemer.cc/resample-imbalanced-data/)\]
* [ ] \(SGD and mini-batch sizes\) Parallelizing Stochastic Gradient Descent for Least Squares Regression: mini-batching, averaging, and model misspecification \[[paper](https://arxiv.org/abs/1610.03774)\]
* [ ] Emergent properties of the local geometry of neural loss landscapes \[[paper](https://arxiv.org/abs/1910.05929)\]
* [ ] Possible BN and Dropout incompatibilities are described in this \[[paper](https://arxiv.org/abs/1801.05134)\]
* [ ] Visual exploration of gaussian processes \[[distill](https://distill.pub/2019/visual-exploration-gaussian-processes/)\]
* Reinforcement Learning Tutorials:
  * [ ] Policy gradient algorithms \[[pg-is-all-you-need](https://github.com/MrSyee/pg-is-all-you-need)\]
  * [ ] A step-by-step tutorial from DQN to Rainbow \[[Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need)\]
* [ ] Behaviour Suite for Reinforcement Learning \[[paper](https://arxiv.org/abs/1908.03568)\]
* [ ] "The Paths Perspective on Value Learning" \(RL subproblem\) \[[article](https://distill.pub/2019/paths-perspective-on-value-learning/)\]
* [ ] "Human in the Loop: Deep Learning without Wasteful Labelling" \[[post](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html)\]
* [ ] Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks \[[paper](https://arxiv.org/abs/1905.11286)\]
* [ ] Selective-Backprop \(e.g. "Accelerating Deep Learning by Focusing on the Biggest Losers"\)
* [ ] Representer Point Selection for Explaining Deep Neural Networks \[[post](https://blog.ml.cmu.edu/2019/04/19/representer-point-selection-explain-dnn/)\]
* [x] "Adversarial Examples Are Not Bugs, They Are Features" \[[distill-discussion](https://distill.pub/2019/advex-bugs-discussion/)\]—\[[post1](http://gradientscience.org/adv/)\] \(post: :white\_check\_mark:, paper and comments: haven't read\)
* [ ] Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet \[[paper](https://arxiv.org/abs/1904.00760)\]
* [ ] Uncertainty Quantification in Deep Learning \[[post](https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)\]
* [ ] Self-Normalizing Neural Networks a.k.a "SELU paper" \[[paper](https://arxiv.org/abs/1706.02515)\] \(SELU: scaled exponential linear unit\)
* [ ] Mathematics of DL \(appendix of Dive into deep learning\) \[[book](http://d2l.ai/chapter_appendix_math/index.html)\]
* [ ] AlphaStar [[post](https://www.deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning)\] \[[paper](https://www.nature.com/articles/s41586-019-1724-z)]

*  **Data Augmentation**

  \[[Contents](things_to_read.md#contents)\]

* Video explaining RandAugment and comparing it to other augmentation methods \[[video](https://youtu.be/Zzt9i3gDueE)\]
* [ ] RandAugment: Practical automated data augmentation with a reduced search space \[[paper](https://arxiv.org/abs/1909.13719)\]
* [ ] AutoAugment: Learning Augmentation Policies from Data \[[paper](https://arxiv.org/abs/1805.09501)\]
* [ ] A survey on Image Data Augmentation for Deep Learning \[[paper](https://link.springer.com/article/10.1186/s40537-019-0197-0)\]

*  **Other interesting papers and blog posts**

[[Contents](things_to_read.md#contents)]

* [ ] Computing Receptive Fields of Convolutional Neural Networks [[post](https://distill.pub/2019/computing-receptive-fields/)]
* [ ] Wave Physics as an Analog Recurrent Neural Network [[paper](https://arxiv.org/abs/1904.12831)] code:[[wavetorch](https://github.com/fancompute/wavetorch)]
* [ ] 3D Ken Burns Effect from a Single Image [[paper](https://arxiv.org/abs/1909.05483)]
* [ ] Understanding UMAP [[post](https://pair-code.github.io/understanding-umap/)]
* [ ] Loss landscapes and surfaces [[website](https://losslandscape.com/knowledge/)]
* [ ] Neural reparameterization [[paper](https://arxiv.org/abs/1909.04240)]
* [ ] Reinforcement Learning, Fast and Slow [[paper](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613%2819%2930061-0)]
* [ ] Neural Turtle Graphics for Modeling City Road Layouts \[[paper](https://arxiv.org/abs/1910.02055)\]
* [ ] Extracting 2D surface with PreMosa \[[PreMosa](https://cblasse.github.io/premosa/example.html)\]
* [ ] Highway nets \[[post](http://people.idsia.ch/~juergen/highway-networks.html)\]—\[[paper1](https://arxiv.org/abs/1507.06228)\]—\[[paper2](https://arxiv.org/abs/1612.07771)\]
* "Compositional Pattern Producing Networks: A Novel Abstraction of Development" \[[paper](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)\]—\[[post1](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/)—[post2](http://blog.otoro.net/2016/06/02/generating-large-images-from-latent-vectors-part-two/)\]

## Force transduction, mechanobiology, and simulating physical systems

[[Contents](things_to_read.md#contents)]

* Active tissue physics:
   * [ ] Bittig T. *et al.* Dynamics of anisotropic tissue growth [[New J.Phys.2008](https://iopscience.iop.org/article/10.1088/1367-2630/10/6/063001)].
   * [ ] Ranft J. *et al.* Fluidization of tissues by cell division and apoptosis [[PNAS2010](https://www.pnas.org/content/107/49/20863)].
   * [ ] Delarue M. *et al.* Stress distributions and cell flows in a growing cell aggregate [[InterfaceFocus2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4213449/)]. 
* Tissue mechanics formalism:
   * [ ] Unified quantitative characterization of epithelial tissue development [[paper](https://elifesciences.org/articles/08519)] <a name="unified-quantitative-boris">(also inference)</a>
   * [ ] A texture tensor to quantify deformations [[paper](https://link.springer.com/article/10.1007%2Fs10035-003-0126-x)]
   * [ ] Colloquium: Mechanical formalisms for tissue dynamics [[paper](https://link.springer.com/article/10.1140%2Fepje%2Fi2015-15033-4)]
   * [ ] Active dynamics of tissue shear flow [[paper](https://iopscience.iop.org/article/10.1088/1367-2630/aa5756)]
* Force inference:
   * [ ] Experimental validation of force inference in epithelia from cell to tissue scale [[paper](https://www.nature.com/articles/s41598-019-50690-3)]
   * [x] DLITE [[paper](https://www.sciencedirect.com/science/article/pii/S0006349519308215)] [[repo](https://github.com/AllenCellModeling/DLITE)]
   * ["Unified quantitative characterization" paper above](#unified-quantitative-boris).
* Physical Simulation:
   * [ ] [VIPER: Volume Invariant Position-based Elastic Rods](https://arxiv.org/abs/1906.05260) (paper talks about a nice way to quantify fiber-based dynamics, i.e. can be applied as a quantification method in many migration and transport problems along curved trajec-s, e.g. F-actin dynamics in cells with curved geom-s,in addition to detailed derivations of the eqn-s)
* [ ] Force networks, torque balance and Airy stress in the planar vertex model of a confluent epithelium [[paper](https://arxiv.org/pdf/1910.10799.pdf)]
* [ ] Optical estimation of absolute membrane potential using fluorescence lifetime imaging [[paper](https://elifesciences.org/articles/44522)]
* [ ] Hydra Regeneration: Closing the Loop with Mechanical Processes in Morphogenesis (Braun E. _et al._) [[paper](https://www.ncbi.nlm.nih.gov/pubmed/29869336)]
* [ ] Electric-induced reversal of morphogenesis in Hydra (Braun E. _et al._) [[paper](https://arxiv.org/abs/1904.03625)]
* [ ] A scalable pipeline for designing reconfigurable organisms [[paper](https://www.pnas.org/content/early/2020/01/07/1910837117)]
* [ ] Bioelectrical domain walls in homogeneous tissues [[paper](https://www.nature.com/articles/s41567-019-0765-4)]
* [ ] "Inferring single-trial neural population dynamics using sequential auto-encoders" (_See_ in RNN section; Neuron firing dynamics approximated as a [dynamical system described by RNNs](#rnn-dynamical-system-nature)).
* [ ] "Up to two billion times acceleration of scientific simulations with deep neural architecture search" [[paper](https://arxiv.org/abs/2001.08055)]

