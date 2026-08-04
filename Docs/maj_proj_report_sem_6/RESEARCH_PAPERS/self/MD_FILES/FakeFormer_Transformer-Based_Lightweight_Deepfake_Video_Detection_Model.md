2025 7th International Conference on Information Science, Electrical and Automation Engineering (ISEAE)
# FakeFormer: Transformer-Based Lightweight Deepfake Video Detection Model



1 [st] Xuefei Wang
_College_ _of_ _Computer_ _Science_ _and_ _Technology_
_Ocean_ _University_ _of_ _China_
Qingdao, China
wangxuefei@stu.ouc.edu.cn



2 [nd] Guoqiang Zhong*
_College_ _of_ _Computer_ _Science_ _and_ _Technology_
_Ocean_ _University_ _of_ _China_
Qingdao, China
gqzhong@ouc.edu.cn



3 [rd] Qiang Song
_Bureau_ _of_ _Natural_ _Resources_
_Qingdao_ _West_ _Coast_ _New_ _Area_
Qingdao, China
sq6364@163.com



_**Abstract**_ **—In recent years, deepfake videos become increasingly**
**realistic,** **making** **them** **nearly** **indistinguishable** **from** **the** **naked**
**eye.** **The** **misuse** **of** **these** **deepfake** **videos** **poses** **a** **major** **threat**
**to** **information** **security,** **leading** **researchers** **to** **develop** **effective**
**deepfake** **video** **detection** **models.** **Video** **temporal** **features** **often**
**contain rich and valuable information for identifying the authen-**
**ticity of videos. Given that transformers have proven highly effec-**
**tive** **at** **modeling** **these** **features,** **researchers** **have** **been** **prompted**
**to** **develop** **transformer-based** **detection** **models.** **However,** **due** **to**
**the** **high-dimensional** **nature** **of** **video** **data,** **the** **efficiency** **of** **such**
**models** **is** **often** **compromised,** **making** **it** **a** **challenging** **problem.**
**In** **this** **paper,** **we** **propose** **a** **specialized** **transformer-based** **model**
**for** **deepfake** **video** **detection,** **called** **FakeFormer.** **FakeFormer**
**is** **a** **lightweight** **video-level** **detection** **model** **that** **maintains** **high**
**efficiency,** **despite** **using** **a** **sequence** **of** **video** **frames** **as** **direct**
**input.** **Additionally,** **FakeFormer** **fully** **integrates** **the** **strengths** **of**
**convolutional** **neural** **networks** **and** **transformers,** **enabling** **it** **to**
**effectively model both the temporal and spatial features of videos**
**to** **accurately** **assess** **their** **authenticity.** **Numerous** **experiments**
**validate** **the** **effectiveness** **of** **FakeFormer,** **including** **intra-dataset**
**evaluation,** **cross-dataset** **evaluation** **and** **ablation** **study.**
_**Index**_ _**Terms**_ **—Deepfake** **detection,** **Spatio-temporal** **feature,**
**Lightweight** **video** **transformer**


I. INTRODUCTION

Deepfake is the technique of swapping source and target
faces to create manipulated images and videos. With the
advancement of deep learning techniques, deepfake techniques
have seen significant breakthroughs. However, the popularity
of deepfakes has also created a hidden danger, leading to an
increasing number of real-world cases suffering from the harm
caused by forged videos. As a result, developing effective
deepfake detection technology has become both a valuable
and urgent need.


Fig. 1. Frames of deepfake videos.


Deepfake video detection is essentially a binary classification task that distinguishes between real and fake videos. Existing methods can be broadly categorized into image-level and
video-level approaches. Image-level methods [1], [2] treat each



video frame independently, analyzing them individually and
aggregating the results for a final judgment. However, these
methods only utilize spatial information, ignoring temporal
inconsistencies in forged videos, as illustrated in Fig. 1. Videolevel detection methods [3], [4] add attention to these inconsistencies, enhancing detection performance. However, the
inherent high dimensionality of video data makes extracting
temporal features a challenging task. Additionally, although
transformers have become the first choice for sequential tasks,
they are inefficient when applied to video data. This problem is
further accentuated by the addition of the high computational
complexity associated with the self-attention operation. At the
same time, using recurrent neural networks [5] suffers from a
lack of detection performance, making it difficult to find an
ideal balance.
To address the aforementioned problems, we propose a
lightweight and effective transformer-based deepfake video
detection model, named FakeFormer. Inspired by a study
exploring the core components of transformers, the authors

[6] discover that the strong performance of transformers is
mainly due to their four-stage architecture, achieved by replacing self-attention with parameter-free pooling operations.
FakeFormer is also based on the four-stage architecture and
further combines the strengths of convolutional neural networks [7] and transformers. It takes continuous video frames
as direct input, effectively and cohesively extracting spatial
features from the video frames and temporal features from the
entire video, leveraging both for deepfake detection. Extensive
experiments on several public datasets demonstrate that our
method effectively detects deepfake videos, resulting in rich
performance advantage over existing methods.


II. THE PROPOSED FAKEFORMER MODEL


_A._ _Motivation_


Transformer’s powerful ability to extract temporal features
aligns well with the needs of deepfake video detection tasks,
driving researchers to develop transformer-based detection
models. However, video data introduce a temporal dimension
to image data and need to be adapted for the application of the
visual transformer. The most intuitive approach is to design a
3D transformer, but we find that this model results in extremely
high computational complexity, making it difficult to train.



979-8-3315-1038-1/25/$31.00 ©2025 IEEE 856


Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:41:50 UTC from IEEE Xplore. Restrictions apply.


FakeFormer block1 FakeFormer block2



Input video
frame sequence



Stage1 Stage2 Stage3 Stage4
Embedding dim: 64 Embedding dim: 128 Embedding dim: 256 Embedding dim: 512



(a) Overal framework of FakeFormer (b) FakeFormer block


Fig. 2. (a) The overall framework of FakeFormer. In FakeFormer, stages [1, 2, 3, 4] consist of [4, 4, 12, 4] blocks, respectively. The embedding dimensions
of each stage are [64, 128, 256, 512]. (b) The structure of the two basic blocks in FakeFormer, which are applied to the first stage as well as to the other
stages, respectively.



We test it with two 3090 GPUs, and it takes several hours
to complete one epoch. Another class of methods employs a
convolutional model to extract spatial features, followed by
a transformer to capture temporal features. These two-stage
approaches are similarly inefficient. To better apply transformer to deepfake video detection, we design a lightweight
video-level detection model FakeFormer, based on the core
component of transformer. The overall architecture is shown in
Fig. 2. FakeFormer follows the four-stage design guideline and
directly uses continuous video frame sequences as input. It can
simultaneously extract the spatio-temporal features from the
video and provide the discrimination results without additional
processing.


_B._ _Embedding_ _Layer_


Although deepfake techniques continue to mature, forged
video frames still contain subtle imperfections that can be
detected by well-trained models. These spatial inconsistencies
are crucial for distinguishing real from fake videos, requiring
the model to effectively capture spatial details. In our preliminary experiments, we find that relying solely on the transformer’s four-stage architecture, results in poor performance in
the deepfake video detection task. And this issue persists even
when fine-tuning with a pre-trained version on a large-scale
dataset. We analyze that it is caused by the model’s inability to
capture spatial details. Subsequently, we find that adding two
simple convolutional layers before the first stage significantly
improves the model’s performance, confirming our hypothesis.
In the preliminary experiments, we use a single video frame
as input, where the input to the embedding layer consists
of image patches. In our approach, we use a sequence of
consecutive video frames as input, where the embedding
layer receives images. Since the original embedding module
is a linear mapping and is no longer suitable, it has been
removed. To compensate for the loss of the original embedding
module and more effectively capture spatial details in the
video frames, we introduce multi-layer convolution before
each stage to perform the embedding operation. Rather than
manually designing the convolutional layers, we employ a



lightweight convolutional model, ResNet34, and divide it into
four stages [64, 128, 256, 512] according to the dimensions
of the resulting feature space, which are inserted before each
corresponding stage of our model.


_C._ _Basic_ _block_


In real videos, the spatial details in the frames remain consistent throughout the sequence, whereas in fake videos, they
show inconsistency. This temporal variation in spatial details is
crucial for deepfake detection. After applying the embedding
operation to the video frames, the resulting feature space
is relatively large. Using self-attention to extract temporal
features requires flattening the feature space, which will lead to
the loss of valuable spatial information and further impact the
extraction of effective temporal inconsistency cues. To address
this, we design a ConvLSTM module to replace self-attention,
enabling more efficient temporal feature extraction from video
frame sequences. ConvLSTM is a two-layer network, with
both the input and hidden dimensions matching the embedding
dimension. It uses a 3x3 convolutional kernel and applies the
hidden state of the second layer at each time step as the
output. In our design, the ConvLSTM module simultaneously
computes the input gate _it_, forget gate _ft_, output gate _ot_, and
candidate state _C_ [˜] _t_ through a single convolutional operation.
Specifically, the input feature _Xt_ and the hidden state _Ht−_ 1
from the previous time step are concatenated to form a new
tensor _Zt_ :


_Zt_ = _Concat_ ( _Xt, Ht−_ 1) _._ (1)


Assuming the number of channels for _Xt_ and _Ht−_ 1 is
_C_, the number of channels for _Zt_ is 2 _C_ . We then apply a
convolutional operation with an input channel number of 2 _C_
and an output channel number of 4 _C_ to generate a tensor _Gt_
with 4 _C_ channels:


_Gt_ = _Conv_ ( _Zt_ ) _._ (2)


Then, _Gt_ is split along the channel dimension into four
groups, each containing _C_ channels, which are used to com


857


Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:41:50 UTC from IEEE Xplore. Restrictions apply.


pute the input gate _it_, forget gate _ft_, output gate _ot_, and
candidate state _C_ [˜] _t_, respectively. The formulas are as follows:

_it_ = _σ_ ( _G_ [0:] _t_ _[C][−]_ [1] ) _,_



_ft_ = _σ_ ( _Gt_ _[C]_ [:2] _[C][−]_ [1] ) _,_

_ot_ = _σ_ ( _G_ [2] _t_ _[C]_ [:3] _[C][−]_ [1] ) _,_
_C_ ˜ _t_ = tanh( _G_ [3] _t_ _[C]_ [:4] _[C][−]_ [1] ) _,_



(3)



We use the Adam optimizer with a learning rate of 1 _e_ _[−]_ [4]

and a batch size of 40. The learning rate is halved every 5
epochs during training, and the total number of epochs is 50.
The loss function is the binary cross-entropy loss. Evaluation
metrics include accuracy (ACC) and the area under the ROC
curve (AUC).


_C._ _Intra-Dataset_ _Evalution_


Intra-dataset evaluation refers to training and testing on
homologous data to assess the model’s fundamental expressive
power, which is its ability to extract forgery clues from
deepfake videos. We perform comparisons on the four forgery
methods in FF++ as well as Celeb-DF, with the results shown
in Table. I. Xception [11] is a generalized vision model and
is not specifically designed for deepfake detection, and it
performs the worst among the evaluated methods. In contrast,
the other models are specifically designed for deepfake video
detection tasks and show varying degrees of performance
advantages over Xception. Furthermore, our model achieves
the best performance on each dataset, which substantiates the
effectiveness of the proposed approach.


TABLE I
**INTRA-DATASET** **EVALUTION** **RESULTS.** WE REPORT THE ACC (%) OF
SEVERAL DEEPFAKE DETECTION METHODS ON THE FACEFORENSICS++
AND CELEB-DF DATASETS. BOLD REPRESENTS THE BEST RESULTS.


Method DF F2F FS NT Celeb-DF
Xception [11] 91 _._ 37 90 _._ 28 91 _._ 54 88 _._ 49 85 _._ 19
CNN-RNN [5] 94.63 92.28 93.15 89.02 91.75
3DCNN [4] 95.88 91.21 92.67 90.30 94.37
ADD [3] 94.12 93.09 94.25 88.77 92.74
RE-C [1] 96.06 95.48 97.33 93.52 95.90
UCT [12] 97.83 97.05 96.68 92.74 96.13
FakeFormer **98** _._ **35** **97** _._ **12** **97** _._ **86** **94** _._ **40** **97** _._ **29**


_D._ _Cross-Dataset_ _Evaluation_


To verify the generalization ability of FakeFormer, we
train it on FF++ and test it on Celeb-DF and DFDC. We
compare the performance of several models, and the results
are presented in Table. II. Although the models have never
seen the data from the two test datasets during training, there
are still some common issues across different deepfake videos,
such as defects in details. Therefore, after training, all the
models maintain varying degrees of generalization ability.
Additionally, it can be seen that our model achieves the best
results, indicating that our model can more effectively mine
the common spatio-temporal inconsistency defects existing in
forged videos, thereby distinguishing between real and fake
videos more efficiently.


_E._ _Ablation_ _Study_


We conduct experiments on Celeb-DF to evaluate the impact
of different temporal feature extraction modules and their
placement on detection performance. The results are shown in
Table. III. The model without any temporal module, relying
solely on the identity operation, performs the worst. Introducing a temporal feature extraction module in the first stage



where 0 : _C −_ 1 represents the first _C_ channels, and the others
follow accordingly.
Finally, the cell state and hidden state are updated based on
the gating signal and candidate state:

_Ct_ = _ft ⊙_ _Ct−_ 1 + _it ⊙_ _C_ [˜] _t,_
(4)
_Ht_ = _ot ⊙_ tanh( _Ct_ ) _,_


where _⊙_ denotes the element-wise multiplication.
The temporal inconsistency required to identify forged
videos is often hidden in the shallow features of the video
frame sequence. However, the higher stages of the model
extract global high-level semantics, such features exhibit relatively high consistency throughout the sequence, regardless
of whether the video is real or fake. Therefore, we use
ConvLSTM only in the first stage of the model to extract
temporal inconsistency cues. We further prove our point by
comparing the positions where ConvLSTM is applied in the
ablation experiments. Finally, for the other stages of the model,
we use identity, a parameter-free operation to reduce the
overall complexity of the model.


III. EXPERIMENT

In this section, we demonstrate the effectiveness of our
method through experimental results, including intra-dataset
evaluation, cross-dataset evaluation, ablation study and visualization experiment.


_A._ _Datasets_

We validate our model using three publicly available
datasets: FaceForensics++ [2], Celeb-DF [8] and DFDC [9].
FaceForensics++ (FF++) contains 1000 real videos from
YouTube and 4000 fake videos generated using four forgery
methods: DeepFakes (DF), FaceSwap (FS), Face2Face (F2F),
and NeuralTextures (NT). FF++ offers three versions with different compression rates (raw, c23, c40). In our experiments,
we use the low-compression version (c23, HQ). Celeb-DF is a
high-quality dataset containing 890 real videos and over 5000
fake videos, which we process using data equalization techniques. DFDC consists of 400GB videos and is a challenging
dataset, filled with various types of noise, making it more
difficult to identify the authenticity of the videos.


_B._ _Implementation_ _Details_

In the data preprocessing phase, we divide each video into
eight segments and select four consecutive frames from each.
We then use the open-source dlib [10] face detector to crop
the face regions and resize each image to 224x224. For each
dataset, we split it into training, validation, and testing sets
with an 8:1:1 ratio.



858


Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:41:50 UTC from IEEE Xplore. Restrictions apply.


TABLE II
**CROSS-DATASET** **EVALUATION** **RESULTS.** ALL THE MODELS ARE TRAINED
ON FF++ THEN TESTED ON DFDC AND CELEB-DF. THE EVALUATION
METRIC IS AUC (%), WITH BOLD INDICATING THE BEST RESULTS.


Test datasets
Methods Training dataset

Celeb-DF DFDC Cross-Avg
Xception [11] FF++ 64.2 59.4 62.30
CNN-RNN [5] FF++ 65.8 60.3 63.05
3DCNN [4] FF++ 67.4 62.8 65.10
ADD [3] FF++ 67.9 61.4 64.65
RE-C [1] FF++ 69.3 66.7 68.00
UCT [12] FF++ 70.5 64.1 67.30
FakeFormer FF++ **72** _._ **1** **67** _._ **8** **69** _._ **95**


significantly enhances the model’s performance. Moreover, the
ConvLSTM module achieves better performance than the selfattention module, indicating that effectively preserving spatial
features facilitates temporal information extraction. Additionally, adding ConvLSTM modules to the other three stages does
not yield better results than using it in the first stage. Finally,
incorporating ConvLSTM in both the first and second stages
leads to a 0.05% performance improvement compared to using
it only in the first stage. However, considering the trade-off
between computational cost and performance, we ultimately
use the ConvLSTM module only in the first stage.


TABLE III
**ABLATION** **STUDY** **RESULTS.** WE REPORT THE ACC (%) ON THE
CELEB-DF DATASETS.

|Temporal module|Adding position|Celeb-DF|
|---|---|---|
|Identity<br>Self-attention<br>ConvLSTM|First stage<br>First stage<br>First stage|94_._92<br>96_._57<br>97_._29|
|ConvLSTM<br>ConvLSTM<br>ConvLSTM<br>ConvLSTM|Second stage<br>Third stage<br>Fourth stage<br>First and second stages|97_._04<br>95_._78<br>95_._13<br>**97**_._**34**|



_F._ _Visualization_


To intuitively illustrate the rationale behind our method’s
judgment in detecting fake videos, we use Grad-CAM [13]
to generate activation maps and present the results using the
FF++ dataset as an example. As shown in Fig. 3, the first
row represents the video frames input to the model, while
the second row shows their corresponding activation maps.
Additionally, the column labeled YouTube represents real
video frames, while the remaining four columns correspond
to the four forgery methods in FF++. It can be seen that the
model accurately identifies forgeries, as indicated by the red
regions in the activation maps, across all four forgery methods.
In contrast, no forged regions are detected in real videos.


IV. CONCLUSION


In this paper, we propose FakeFormer, a lightweight and
effective video transformer model for deepfake detection.
FakeFormer leverages the core strengths of convolutional neural networks and transformers. It takes a continuous sequence
of video frames as the direct input, fully extracting valuable



YouTube DeepFakes Face2Face FaceSwap NeuralTextures


Fig. 3. Visualization results of the Grad-CAM outputs.


temporal and spatial features to produce authenticity identification results for the video. Experiments demonstrate that
FakeFormer outperforms most existing methods for deepfake
video detection. In future work, we will focus on improving
its generalization to address potential future forgeries.


V. ACKNOWLEDGMENTS


This work was partially supported by the National Natural Science Foundation of China (NSFC) under Grants No.
42476194 and No. U24A20242, the Natural Science Foundation of Shandong Province under Grants No. ZR2021ZD19
and No. ZR2024MF097, and Project of Associative Training
of Ocean University of China under Grant No. 202265007.
We want to thank “Qingdao AI Computing Center” and “EcoInnovation Center” for providing inclusive computing power
and technical support of MindSpore during the completion of
this paper.


REFERENCES


[1] J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, and X. Yang, “End-toend reconstruction-classification learning for face forgery detection,” in
_CVPR_, 2022, pp. 4113–4122.

[2] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and
M. Nießner, “Faceforensics++: Learning to detect manipulated facial
images,” in _ICCV_, 2019, pp. 1–11.

[3] B. Zi, M. Chang, J. Chen, X. Ma, and Y.-G. Jiang, “Wilddeepfake: A
challenging real-world dataset for deepfake detection,” in _ACM_ _MM_,
2020, pp. 2382–2390.

[4] D. Zhang, C. Li, F. Lin, D. Zeng, and S. Ge, “Detecting deepfake videos
with temporal dropout 3DCNN.” in _IJCAI_, 2021, pp. 1288–1294.

[5] E. Sabir, J. Cheng, A. Jaiswal, W. AbdAlmageed, I. Masi, and P. Natarajan, “Recurrent convolutional strategies for face manipulation detection
in videos,” _Interfaces_ _(GUI)_, vol. 3, no. 1, pp. 80–87, 2019.

[6] W. Yu, M. Luo, P. Zhou, C. Si, Y. Zhou, X. Wang, J. Feng, and S. Yan,
“Metaformer is actually what you need for vision,” in _CVPR_, 2022, pp.
10 819–10 829.

[7] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in _CVPR_, 2016, pp. 770–778.

[8] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A large-scale
challenging dataset for DeepFake forensics,” in _CVPR_, 2020, pp. 3204–
3213.

[9] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and
C. C. Ferrer, “The deepfake detection challenge (dfdc) dataset,” _arXiv_
_preprint_ _arXiv:2006.07397_, 2020.

[10] D. E. King, “Dlib-ml: A machine learning toolkit,” _J. Mach. Learn. Res._,
vol. 10, pp. 1755–1758, 2009.

[11] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in _CVPR_, 2017, pp. 1251–1258.

[12] B. Yu, W. Li, X. Li, J. Zhou, and J. Lu, “Uncertainty-aware hierarchical
labeling for face forgery detection,” _PR_, vol. 153, p. 110526, 2024.

[13] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and
D. Batra, “Grad-cam: Visual explanations from deep networks via
gradient-based localization,” in _ICCV_, 2017, pp. 618–626.



859


Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:41:50 UTC from IEEE Xplore. Restrictions apply.


