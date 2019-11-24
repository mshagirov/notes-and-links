# Papers, Blog Posts and Other Resources:
---
[[Main page](README.md)]
## Contents

- [Image Analysis and Segmentation](#image-analysis-and-segmentation)
 - [Deep Image Priors (DIPs)](#deep-image-priors)
- [Graph Neural Networks](#graph-neural-networks)
- [Generative models](#generative-models)
- [General Deep Learning, and RL](#general-deep-learning-ml-and-rl)
- [Data Augmentation](#data-augmentation)
- [Other Interesting Papers in ML](#other-interesting-papers-and-blog-posts)
- [Force transduction, and mechanobiology](#force-transduction-and-mechanobiology)
---
## Image Analysis and Segmentation
* [ ] EfficientDet: Scalable and Efficient Object Detection [[paper](https://arxiv.org/abs/1911.09070)]
* [ ] Differentiable Mask-Matching Network (DMM-net)
[[repo](https://github.com/ZENGXH/DMM_Net)]
[[paper](https://www.cs.toronto.edu/~xiaohui/dmm/paper/dmmnet_iccv19.pdf)]
* [ ] Panoptic segmentation
[[paper](https://arxiv.org/abs/1801.00868)] (_see detectron2 for implementation_)
* [ ] Faster R-CNN
[[paper](https://arxiv.org/abs/1506.01497)]
[[pytorch](https://github.com/ZENGXH/faster-rcnn.pytorch)]
* [ ] Mask R-CNN
[[paper](https://arxiv.org/abs/1703.06870)]
[[repo](https://github.com/facebookresearch/maskrcnn-benchmark)]
* [ ] R-FCN (Region-based Fully Convolutional Networks)
[[paper](https://arxiv.org/abs/1605.06409)]
* [ ] Detectron (uses Mask R-CNN and others above, w/ _new ver. detectron2_)
[[repo](https://github.com/facebookresearch/Detectron)]
[<img src="https://github.com/facebookresearch/detectron2/blob/master/.github/Detectron2-Logo-Horz.svg" alt="detectron2" width="100"/>](https://github.com/facebookresearch/detectron2/blob/master/README.md)
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
* _Related to_ [DIPs](#deep-image-priors), 
* [ ] High-Quality Self-Supervised Deep Image Denoising [[repo w/ paper](https://github.com/NVlabs/selfsupervised-denoising)]
* [ ] Noise2Self [[paper](https://arxiv.org/abs/1901.11365)] and [[repo](https://github.com/czbiohub/noise2self)]
* [ ] pN2V [[PN2V repo](https://github.com/juglab/pn2v)] might help to revise [[N2V repo](https://github.com/juglab/n2v)] and
[[N2N repo](https://github.com/NVlabs/noise2noise)] [[paper](https://arxiv.org/abs/1803.04189)]
* [ ] On Network Design Spaces for Visual Recognition [[paper](https://arxiv.org/abs/1905.13214)] [[repo](https://github.com/facebookresearch/pycls)]

### Deep Image Priors
* [ ] Deep Image Prior
[[website](https://dmitryulyanov.github.io/deep_image_prior)]
* [ ] Double DIP
[[website](http://www.wisdom.weizmann.ac.il/~vision/DoubleDIP/)]
[[repo](https://github.com/yossigandelsman/DoubleDIP)] might need
[["Blind dehazing"](https://github.com/YuvalBahat/Dehazing-Airlight-estimation)]

## Graph Neural Networks
* [ ] A Comprehensive Survey on Graph Neural Networks [[paper](https://arxiv.org/abs/1901.00596)]
* [ ] Neural Message Passing for Quantum Chemistry [[paper](https://arxiv.org/abs/1704.01212)]

## Generative models
* [ ] Continual Unsupervised Representation Learning [[paper](https://arxiv.org/abs/1910.14481)][[repo](https://github.com/deepmind/deepmind-research/tree/master/curl)]
* [ ] Learning Implicit Generative Models by Matching Perceptual Features [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/dos_Santos_Learning_Implicit_Generative_Models_by_Matching_Perceptual_Features_ICCV_2019_paper.html)]
* [ ] GAN [[paper](https://arxiv.org/abs/1406.2661)]
* [ ] GAN hacks [[post](https://github.com/soumith/ganhacks)]
* [ ] Seeing What a GAN Cannot Generate [[paper](https://arxiv.org/pdf/1910.11626.pdf)]
* [ ] SinGAN: Learning a Generative Model from a Single Natural Image [[paper](https://arxiv.org/abs/1905.01164)]
* [ ] On Adversarial Mixup Resynthesis [[video presentation](https://www.youtube.com/watch?v=ezbC3_VZeNY)] [[paper](https://arxiv.org/abs/1903.02709)]
* [ ] Conditional GAN [[paper](https://arxiv.org/abs/1411.1784?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=77587488&_hsenc=p2ANqtz--i5nQIm7lOwKMygW3rZvR9W1dgbq-yKtBIuLO0OdAbVFexTcWQvh6d5jHGk0Fj2Et8vhqTYcnuCs9ITplGKwlHIvmXag&_hsmi=77587488)]
* [ ] CycleGAN [[website](https://junyanz.github.io/CycleGAN/)]
* [ ] StarGAN [[paper](https://arxiv.org/abs/1711.09020)] [[repo](https://github.com/yunjey/StarGAN)]
* [ ] PixelCNN++ [[paper](https://openreview.net/pdf?id=BJrFC6ceg)] [[repo](https://github.com/openai/pixel-cnn)]
* [ ] PixelVAE++ [[paper](https://arxiv.org/abs/1908.09948)] [[PixelVAE](https://arxiv.org/abs/1702.08658)] [repo?]
* [ ] Progressive Growing of GANs for Improved Quality, Stability, and Variation [[paper](https://arxiv.org/abs/1710.10196)]
* [ ] Few-shot Video-to-Video Synthesis [[website with links](https://nvlabs.github.io/few-shot-vid2vid/)]
* [ ] WaveNet [[website](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)]
* "Generating Large Images from Latent Vectors" _see_ "Compositional Pattern Producing Networks" in "General Deep Learning" section.
* _See_ Attention (Transformers) in ["General Deep Learning"](#general-deep-learning-ml-and-rl) section.

## General Deep Learning, ML and RL
* [ ] See abstracts and references in [http://bayesiandeeplearning.org](http://bayesiandeeplearning.org) (2016-19 workshops)
* [ ] Self-training with Noisy Student improves ImageNet classification [[paper](https://arxiv.org/abs/1911.04252)]
* [ ] Meta-Learning Deep Energy-Based Memory Models [[paper](https://arxiv.org/abs/1910.02720)]
* [ ] Why re-sampling imbalanced data isn’t always the best idea [[post](https://stroemer.cc/resample-imbalanced-data/)]
* [ ] (Model discovery) Data-driven discovery of coordinates and governing equations [[paper](https://www.pnas.org/content/116/45/22445)]
* [ ] DiffTaichi: Differentiable Programming for Physical Simulation [[paper](https://arxiv.org/abs/1910.00935)]
* [ ] (SGD and mini-batch sizes) Parallelizing Stochastic Gradient Descent for Least Squares Regression: mini-batching, averaging, and model misspecification [[paper](https://arxiv.org/abs/1610.03774)]
* [ ] Self-supervised representation learning (by Lilian Weng) [[post](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)]
* [ ] Neocortical plasticity (about unsupervised learning in neocortex)[[paper](https://openreview.net/pdf?id=S1g_N7FIUS)]
* [ ] An intriguing failing of convolutional neural networks and the CoordConv solution
* [ ] On the adequacy of untuned warmup for adaptive optimization [[paper](https://arxiv.org/abs/1910.04209)]
* [ ] Distilling the Knowledge in a Neural Network [[paper](https://arxiv.org/abs/1503.02531)]
* [x] "A recipe for training neural networks" by A. Karpathy [[post](http://karpathy.github.io/2019/04/25/recipe/)]
* [ ] Weight Uncertainty in Neural Networks [[paper](https://arxiv.org/abs/1505.05424)]
* [ ] "The Building Blocks of Interpretability" [[distill](https://distill.pub/2018/building-blocks/)]
* [ ] "Bias and Generalization in Deep Generative Models" by Zhao *et al.* [[post](https://ermongroup.github.io/blog/bias-and-generalization-dgm/)]
* [ ] "Exploring Neural Networks with Activation Atlases" [[distill](https://distill.pub/2019/activation-atlas/)]
* [ ] Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly [[paper](https://arxiv.org/abs/1903.06694)]
* Bias-Resilient Neural Network [[paper](https://arxiv.org/abs/1910.03676)]
* [ ] Bayesian Active Learning (Baal) (from ElementAI) [[repo with links](https://github.com/ElementAI/baal)]—[[docs with a list of bayesian active learning papers](https://baal.readthedocs.io/en/latest/)].
* [ ] Emergent properties of the local geometry of neural loss landscapes [[paper](https://arxiv.org/abs/1910.05929)]
* [ ] Possible BN and Dropout incompatibilities are described in this [[paper](https://arxiv.org/abs/1801.05134)]
* [ ] Visual exploration of gaussian processes [[distill](https://distill.pub/2019/visual-exploration-gaussian-processes/)]
* Recurrent Neural Networks (RNNs)
  * [ ] "The Unreasonable Effectiveness of Recurrent Neural Networks" by A. Karpathy [[post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)]
  * [ ] "Visualizing memorization in RNNs" [[distill](https://distill.pub/2019/memorization-in-rnns/)]
* [ ] __Mish__: Self Regularized Non-Monotonic Activation Function [[Repo with results](https://github.com/digantamisra98/Mish)]—[[on fast.ai forums](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299)] (this seems to perform better than ReLU)
* Attention ((Transformers):
  * [ ] Transformers: State-of-the-art Natural Language Processing [[review](https://arxiv.org/abs/1910.03771)]
  * [ ] Tool for analysing transformers [[exBERT](http://exbert.net/)]
  * [ ] Attention is all you need [[paper](https://arxiv.org/abs/1706.03762)]
  * [ ] Transformer-XL [[paper](https://arxiv.org/abs/1901.02860)]
  * [ ] Write With Transformer (GTP2-XL) [[website](https://transformer.huggingface.co)] 
  * [ ] NLP papers from hugging face [[website](https://huggingface.co)]
* [ ] Image transformer (related to Transformer nets in NLP) [[paper](https://arxiv.org/abs/1802.05751)]
* [ ] Stabilizing Transformers for Reinforcement Learning [[paper](https://arxiv.org/abs/1910.06764)]
* [ ] Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks [[paper](https://arxiv.org/abs/1910.01663)]
* [ ] Domain-Adversarial Training of Neural Networks (Ganin Yaroslav, _et al._) [[paper1](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)]
* [ ] "Domain Randomization for Sim2Real Transfer" by L. Weng [[post](https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html)]
  * [ ] "Learning Dexterity"(OpenAI) [[post](https://openai.com/blog/learning-dexterity/)]
  * [ ] "MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies" [[website](https://xbpeng.github.io/projects/MCP/)]
* [ ] Adapting Pretrained Representations to Diverse Tasks [[transfer learning](https://arxiv.org/pdf/1903.05987.pdf)]
* [ ] [World Models-- Can agents learn inside of their own dreams?](https://worldmodels.github.io)
* Reinforcement Learning Tutorials:
  * [ ] Policy gradient algorithms [[pg-is-all-you-need](https://github.com/MrSyee/pg-is-all-you-need)]
  * [ ] A step-by-step tutorial from DQN to Rainbow [[Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need)]
* [ ] Behaviour Suite for Reinforcement Learning [[paper](https://arxiv.org/abs/1908.03568)]
* [ ] "The Paths Perspective on Value Learning" (RL subproblem) [[article](https://distill.pub/2019/paths-perspective-on-value-learning/)]
* [ ] "Human in the Loop: Deep Learning without Wasteful Labelling" [[post](https://oatml.cs.ox.ac.uk/blog/2019/06/24/batchbald.html)]
* [ ] Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks [[paper](https://arxiv.org/abs/1905.11286)]
* [ ] Selective-Backprop (e.g. "Accelerating Deep Learning by Focusing on the Biggest Losers")
* [ ] Representer Point Selection for Explaining Deep Neural Networks [[post](https://blog.ml.cmu.edu/2019/04/19/representer-point-selection-explain-dnn/)]
* [x] "Adversarial Examples Are Not Bugs, They Are Features" [[distill-discussion](https://distill.pub/2019/advex-bugs-discussion/)]—[[post1](http://gradientscience.org/adv/)] (post: :white_check_mark:, paper and comments: haven't read)
* [ ] Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet [[paper](https://arxiv.org/abs/1904.00760)]
* [ ] Uncertainty Quantification in Deep Learning [[post](https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)]
* [ ] Self-Normalizing Neural Networks a.k.a "SELU paper" [[paper](https://arxiv.org/abs/1706.02515)] (SELU: scaled exponential linear unit)
* [ ] Mathematics of DL (appendix of Dive into deep learning) [[book](http://d2l.ai/chapter_appendix_math/index.html)]
## Data Augmentation
* Video explaining RandAugment and comparing it to other augmentation methods [[video](https://youtu.be/Zzt9i3gDueE)]
* [ ] RandAugment: Practical automated data augmentation with a reduced search space [[paper](https://arxiv.org/abs/1909.13719)]
* [ ] AutoAugment: Learning Augmentation Policies from Data [[paper](https://arxiv.org/abs/1805.09501)]
* [ ] A survey on Image Data Augmentation for Deep Learning [[paper](https://link.springer.com/article/10.1186/s40537-019-0197-0)]
## Other interesting papers and blog posts
* [ ] Computing Receptive Fields of Convolutional Neural Networks [[post](https://distill.pub/2019/computing-receptive-fields/)]
* [ ] 3D Ken Burns Effect from a Single Image [[paper](https://arxiv.org/abs/1909.05483)]
* [ ] Understanding UMAP [[post](https://pair-code.github.io/understanding-umap/)]
* [ ] Loss landscapes and surfaces [[website](https://losslandscape.com/knowledge/)]
* [ ] AlphaStar [[post](https://www.deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning)] [[paper](https://www.nature.com/articles/s41586-019-1724-z)]
* [ ] Neural reparameterization [[paper](https://arxiv.org/abs/1909.04240)]
* [ ] Reinforcement Learning, Fast and Slow [[paper](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(19)30061-0)]
* [ ] Neural Turtle Graphics for Modeling City Road Layouts [[paper](https://arxiv.org/abs/1910.02055)]
* [ ] Extracting 2D surface with PreMosa [[PreMosa](https://cblasse.github.io/premosa/example.html)]
* [ ] Highway nets [[post](http://people.idsia.ch/~juergen/highway-networks.html)]—[[paper1](https://arxiv.org/abs/1507.06228)]—[[paper2](https://arxiv.org/abs/1612.07771)]
* "Compositional Pattern Producing Networks: A Novel Abstraction of Development" [[paper](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)]—[[post1](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/)—[post2](http://blog.otoro.net/2016/06/02/generating-large-images-from-latent-vectors-part-two/)]

## Force transduction, and mechanobiology
* [ ] Experimental validation of force inference in epithelia from cell to tissue scale [[paper](https://www.nature.com/articles/s41598-019-50690-3)]
* [ ] Optical estimation of absolute membrane potential using fluorescence lifetime imaging [[paper](https://elifesciences.org/articles/44522)]
* [x] DLITE [[paper](https://www.sciencedirect.com/science/article/pii/S0006349519308215)] [[repo](https://github.com/AllenCellModeling/DLITE)]
* [ ] Force networks, torque balance and Airy stress in the planar vertex model of a confluent epithelium [[paper](https://arxiv.org/pdf/1910.10799.pdf)]

