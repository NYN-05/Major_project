Received 12 January 2026, accepted 19 January 2026, date of publication 28 January 2026, date of current version 5 February 2026.


_Digital Object Identifier 10.1109/ACCESS.2026.3659021_

# A Quantum-Hybrid Framework for Enhanced Deepfake Detection: Integrating QAOA-Based Feature Selection With Quantum-Inspired Attention Mechanisms


FARHAAN KHAN, ADITYA SAREEN, AKASH SURESH KUMAR, AND M. BHUVANESWARI
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu 632014, India


Corresponding author: M. Bhuvaneswari (m.bhuvaneswari@vit.ac.in)


**ABSTRACT** Deepfake technology has created major new challenges to the authentication of digital media and
the security of digital information. The traditional methods using deep learning to detect deepfakes have shown
good results. However, many of these traditional methods suffer from redundant features, imbalanced classes,
and poor interpretability. This paper introduces a new hybrid quantum-classical framework for detecting
deepfakes that combines InceptionResnetV1 to extract high dimensional representation with a Convolutional
Neural Network (CNN), and a Quantum Inspired Attention Mechanism to select the most important features
to process. Additionally, this paper proposes a Quantum Approximate Optimization Algorithm (QAOA) for
feature selection based on high dimensional representations of images obtained from an InceptionResnetV1.
QAOA selects the most informative features to be input into the CNN and helps reduce dimensionality while
maintaining the ability to discriminate between classes of data. The selected features are then passed to a
Multi-Head Quantum Inspired Attention Layer that utilizes quantum interference patterns to re-weight feature
importance dynamically. The use of a focal Loss function with label smoothing and confidence penalty
is also proposed to balance the class distribution in the deepfake datasets. Experiments conducted on the
FaceForensics++ benchmark dataset show that the quantum-hybrid method outperforms classical methods
in terms of accuracy, precision, recall, and calibration metrics. Grad-CAM is used as a visualization tool
to provide interpretable output by showing which facial areas the model is focusing on when identifying
forensic evidence.


**INDEX TERMS** Deepfake detection, quantum computing, quantum approximate optimization algorithm,
attention mechanisms, feature selection, multimedia forensics, computer vision, hybrid quantum-classical
systems.



**I.** **INTRODUCTION**
The advancements in Artificial Intelligence (AI) and Deep
Learning have accelerated the development of highly advanced
forms of synthetic media also commonly known as DeepFakes;
where AI can create videos and images using deep learning
models which can convincingly alter facial expressions, swap
identities and fabricate entirely new content that cannot be
distinguished from real content when viewed by a human.


The associate editor coordinating the review of this manuscript and


approving it for publication was Sotirios Goudos .



Although deepfake technology has many positive uses in
the areas of entertainment, education, artistic expression
etc., the negative consequences of deepfake misuse are far
reaching and pose serious threats to individuals right to privacy,
democratic process, judicial systems and overall trust in digital
information. The ability to use deepfakes for the purposes
of spreading misinformation, committing fraud, creating
non-consensual pornography and destroying the reputation
of public figures requires the development of methods
capable of reliably detecting and identifying such manipulated
content.



2026 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License.
VOLUME 14, 2026 For more information, see https://creativecommons.org/licenses/by/4.0/ 17853


Traditional methods for deepfake detection rely heavily on
Convolutional Neural Networks (CNNs) and their variants.
Such networks have shown significant success in identifying
manipulation artifacts through hierarchical feature extraction.
Many state-of-the-art architectures including EfficientNet,
Xception and various ResNet variants have been widely
used for this purpose. Nevertheless, traditional methods for
deepfake Detection suffer from a number of fundamental
limitations. Firstly, CNNs produce large amounts of data from
the high-dimensional feature space of the input data which
can include redundant and non-discriminative data. This
redundancy contributes to inefficient computation time
and possible model over-fitting. Secondly, most deepfake
datasets suffer from an extreme class imbalance issue, where
manipulated samples often greatly outnumber authentic
samples. This leads to biased model predictions based on the
class imbalance of the training dataset. Finally, the ‘‘closed
box’’ nature of the deep neural network makes it difficult to
understand how different parts of the face contribute to the
final classification decision. This lack of transparency makes
it difficult to apply deepfake Detection Methods in forensic
applications.
Recently, advancements in Quantum Computing and
Quantum-Inspired Algorithms have provided new alternatives
to address the computational issues associated with Machine
Learning. Quantum Computing utilizes principles such as
superposition, entanglement, and quantum interference to
solve certain problems exponentially faster than Classical
Computers. Although Large-Scale Fault-Tolerant Quantum
Computers have yet to be developed, classical algorithms
inspired by quantum computation have already shown
promise across a variety of optimization and machine
learning problems. One such algorithm is the Quantum
Approximate Optimization Algorithm (QAOA), which was
originally proposed in the proposed research work. QAOA
reformulates combinatorial optimization problems as the
task of finding the ground states of Quantum Hamiltonians.
As such, QAOA offers potential advantages for feature
selection, hyperparameter optimization, and other discrete
optimization scenarios relevant to machine learning.


_A._ _MOTIVATION AND RESEARCH GAPS_
While a new frontier in artificial intelligence is rapidly
emerging in the field of deepfakes, it represents an almost
entirely untapped opportunity for innovative development in
conjunction with quantum computing.
Traditional methods of detecting deepfakes face 3 primary
limitations, which quantum-inspired methods are capable of
addressing:


1) HIGH-DIMENSIONAL SPACE
Traditional CNN models use a dimension range of 512 to
2048 when extracting feature representations from images.
Many of these dimensions have redundancy and/or low
discriminative capability. The traditional method of selecting
features relies on either heuristic-based methods or greedy



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


algorithms that may converge to suboptimal solutions. The
quantum inspired optimization techniques can allow us to
explore the exponentially large space of feature subsets using
quantum superposition and interference.


2) RELATIONSHIP COMPLEXITY BETWEEN FACIAL FEATURES
Deepfakes create subtle, non-linear relationships between
facial features that classical attention mechanisms have
difficulty capturing. Using quantum inspired attention to
represent features as quantum states with complex valued
amplitudes and phase can provide interference patterns and
correlations like entanglement unavailable to real valued
classical representations.


3) FORENSIC INTERPRETATION
The Forensic application requires both accurate predictions
and transparent reasoning. The quantum framework has inherent interpretability through quantum probability distributions,
quantifying feature importance through Hamiltonian weight,
and visualizing quantum interference patterns to provide
unprecedented insight into how the model makes decisions.


_B._ _CONTRIBUTIONS_
The quantum hybrid framework developed in this study has
been implemented to detect deepfakes using a total of five
innovations that resolve the above-described issues:
1) **QAOA-Based** **Intelligent** **Feature** **Selection:** The
authors have used QAOA to carry out an intelligent
selection of features based on high dimensional
representation generated from pre-trained CNN models.
The feature selection was defined as a combinatorial
optimization problem in order to select the best subset
of features that are most discriminant and less redundant.
Each feature is associated with a qubit; therefore, the
combinatorial optimization is carried out by means of
the QAOA algorithm. The selection process is driven
by a cost Hamiltonian whose terms depend on both the
relevance scores of the features and the penalties due to
correlations between them.
2) **Quantum** **inspired** **multi-head** **attention** **mecha-**
**nism:** A novel multi-head quantum inspired attention
mechanism is proposed by the authors that assigns
complex-valued coefficients to features, and attention
weights are computed according to Born’s rule of
quantum mechanics. This allows the model to capture
subtle correlations between features and interference
patterns that cannot be captured by classical attention
mechanisms.
3) **Balanced** **focal** **loss** **with** **calibration:** The authors
propose a new balanced focal loss function to deal with
the class imbalance present in the datasets used in the
study. This function includes four different types of
weighting: the first is focal weighting, the second is
positive class emphasis, the third is label smoothing and
the fourth is confidence calibration. The use of these
four weighting functions ensures robust training on



17854 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


unbalanced datasets while maintaining well calibrated
estimates of probabilities.
4) **Enhanced** **interpretability** **framework:** The authors
provide a complete interpretability of their results by
integrating gradient-weighted class activation mapping
(Grad-CAM) with the visualization of the quantum
analysis. They do this to obtain a full interpretation
of the results obtained through the heatmaps of the
spatial attention, the distribution of the importance of
the features, the probability patterns of the quantum and
the visualization of the attention weights.
5) **Comprehensive experimental validation:** The authors
performed an exhaustive experimentation with the
FaceForensics++ benchmark, demonstrating better
performances than classical baselines on all the
considered metrics such as Accuracy, Precision, Recall,
F1 score, AUC ROC and Expected Calibration Error
(ECE). Moreover, they provided a thorough ablation
study to quantify the contribution of each individual
component of the quantum-hybrid model.


In the implementation of the experimental methodology,
the authors employed the FaceForensics++ benchmark
dataset that contains multiple manipulation techniques
among which there are: Face2Face, FaceSwap, Deepfakes
and NeuralTextures. The authors have developed a very
complete preprocessing pipeline including extraction of
frames, MTCNN (multi-task cascaded convolutional network)
based face detection, laplacian variance filtering for blur
removal and a wide range of strategies of data augmentation.
The authors compare the performance of the quantum-hybrid
model with respect to the classical InceptionResnetV1 baseline
over several performance metrics. Finally, the authors have
developed an interactive Gradio-based demonstration interface
to allow users to evaluate the model and compare it with other
models in real time.


_C._ _PAPER ORGANIZATION_
In order to provide a comprehensive and unified review of our
quantum-hybrid deepfake detection framework, the remainder
of this paper is organized as follows. Section II presents
related work on deepfake detection using both deep learning
techniques and alternative approaches, quantum computing
applications in machine learning, and attention mechanisms.
This discussion establishes the theoretical foundation and
empirical motivation for our proposed framework.
Section III describes the overall architecture of the
Quantum-Hybrid Deepfake Detection Framework, detailing
the integration of all components that constitute the end-to-end
system.
Section IV explains how we implemented the proposed
framework’s methodology. It includes the pre-processing
procedures applied to each dataset, the classical baseline
architecture, the formulation of the quantum–hybrid model,
and the training protocol followed for all models.



Section V describes the experimental configuration in
terms of the selected evaluation metrics and the development
environment in which all experiments were conducted.
Section VI presents and analyzes the experimental results,
comparing the quantitative performance of the classical and
quantum-hybrid models as well as providing qualitative
assessments of their interpretability.
Section VII discusses the broader implications of our
findings, practical considerations for real-world deployment
of the proposed system, limitations of the current approach,
and potential avenues for future research.
Finally, Section VIII summarizes the key contributions
of the research work and highlights the wider impact of
our research on multimedia forensics and digital media
authentication.


**II.** **RELATED WORK**
The three connected research fields we will be using to create
a hybrid-quantum approach are covered by this section of the
literature review. We reviewed the most recent approaches
to detecting classical deepfakes as well as machine learning
with quantum computing, and the mechanisms of attention
in machine learning. This is necessary to understand the
development, accomplishments and constraints of previous
research in each of these areas; and to give context to our
contributions as well as to identify the gap(s) in the previous
research that our proposed architecture addresses.


_A._ _CLASSICAL DEEPFAKE DETECTION APPROACHES_
Quantum-hybrid deepfake detection represents an important
emerging research direction. It aimed to address the persistent
limitations of current multimedia forensic frameworks.
Deepfake detection has been an active research area since
the early 2010s. The researchers used various approaches
relying on identifying low-level manipulation artifacts such as
compression inconsistencies, color distribution irregularities,
and disruptions in temporal continuity. However, quickly they
became insufficient as deepfake generation techniques grew
more sophisticated.
Deep learning has contributed greatly to the advancement
of this area, and CNNs based on the convolutional layer have
achieved much higher success than previously possible due to
their ability to automatically learn hierarchical features from
data. The development of the FaceForensics++ dataset [1]
by Rossler et al., has provided researchers with a robust and
comprehensive resource for the testing of various techniques
used for detecting deepfakes using a variety of different types
of manipulation.
EfficientNet models [20] were built on the principle of
scaling depth, width, and resolution, to provide high levels of
accuracy at lower computational cost; and InceptionNeXt [2]
uses a combination of inception modules and ConvNeXt to
extract multi-scale features through four parallel channels.
The residual connections [21] have been used to improve the
gradient flow between layers and enhance overall performance
for facial analysis tasks. Researchers have also utilized transfer



VOLUME 14, 2026 17855


learning methods [6] and [7] to leverage pre-trained models
(such as VGGFace2, VGGFace16), to improve both the
efficiency of the training time and accuracy of the detection
method.
Additionally, temporal modeling methods have also had an
important impact. For example, Li et al. [14], have employed
optical flow to identify temporal inconsistencies in manipulated video streams. Furthermore, Sathwik Reddy et al. [15]
used a convolutional neural network and a recurrent deep
learning neural network to detect deepfake videos. The author
used motion patterns, motion continuity, variations in spatial
and temporal features to detect deepfake data.
Petmezas et al. [16] have used a convolutional neural
network and LSTM transformer method to capture and
analyse the frame-level and pixel-level inconsistentcy in
spatial and temporal patterns. It has been used to detect
deepfake sequences. Other examples of temporal modeling
approaches include head pose inconsistency detection [17].
A model attribution and face swap deepfake identification
system was proposed in FAME [18]. It used facial feature
artifacts, frame-level motion variations, and appearance data
to find deepfake information. Capsule-forensics [19] using
capsule networks to better capture spatial relationships.
However, despite these advances, there are still significant
challenges that exist. First, high dimensional feature representation can lead to redundant information being included in
the model resulting in increased computational requirements
and a greater chance of overfitting. Second, class imbalance
remains a common problem as many existing datasets have
a large disparity between the number of real and fake
samples, resulting in a bias toward predicting one class
over the other. Finally, the lack of interpretability of deep
neural networks represents a significant barrier to forensic
acceptance, as experts would like to be able to understand
clearly how a model makes its decision regarding which facial
regions or attributes led to the prediction of a particular class,
in order to support investigations and litigation.
Our proposed hybrid quantum approach is specifically
designed to overcome the identified limitations and will do
so using quantum inspired attention mechanisms, enhanced
interpretability frameworks, and intelligent feature selection
strategies.


_B._ _QUANTUM COMPUTING IN MACHINE LEARNING_
The emergence of quantum computing represents an information processing paradigm shift from classical methods.
It utilizes quantum mechanics principles, quantum computing
can accomplish computations that are beyond the capabilities
of all classical machines. Quantum machine learning
(QML), [22] explains the relationship between quantum
computing and artificial intelligence. It shows that the quantum
algorithms can accelerate machine learning tasks or provide
new computational strategies.
Even though there are numerous theoretical advances
suggesting that quantum systems may significantly outperform



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


classical computers for problems such as quantum sampling,
quantum linear algebra and certain optimization problems,
the use of practical QML algorithms on near-term quantum
hardware, [12], continues to be very difficult due to the
limitations of the hardware. The limitations of the hardware
include the noise, limited coherence times of the Qubits, and
limited circuit depths which make it difficult to achieve largescale QML applications.
One of the most notable examples of a hybrid quantumclassical method for achieving high-performance applications
using near-term quantum devices is the Quantum Approximate
Optimization Algorithm (QAOA). QAOA was proposed by
Farhi et al. [4] and utilizes parameterized quantum circuits
to approximately compute the ground state of quantum
Hamiltonian representations of combinatorial optimization
problems. The classical optimizer then uses the measurement
outcomes to iteratively update the parameters of the quantum
circuits and improve the solution quality. QAOA has been
applied to many types of combinatorial optimization problems,
including but not limited to, MaxCut, Graph Coloring,
Portfolio Optimization and Vehicle Routing, and has also been
applied to machine learning tasks, including feature selection,
which is an inherently combinatorial problem, well-suited to
the structure of QAOA.
Mounika et al. [13] explained the way how important data
were selected. It pruned superfluous information and derived
an optimal quantum kernel. It showed that classical data
can be processed quantum mechanically via suitable feature
mappings. Similarly, Devadas and Sowmya [22] provided a
Survey of Quantum Machine Learning, outlining both the
opportunities and foundational constraints of the field.
Currently, fully fault-tolerant quantum computers that can
provide practical quantum advantage continue to be developed.
However, classical algorithms that emulate key quantum
computational principles on classical hardware have already
shown promise. These classical algorithms utilize structures
such as tensor networks, amplitude encoded representations
and quantum-inspired optimization landscapes. Our work
follows this philosophy and combines classical simulation of
QAOA with conventional optimization techniques to perform
feature selection; allowing us to deploy our methods on current
computational infrastructure.


_C._ _ATTENTION MECHANISMS AND QUANTUM-INSPIRED_
_ATTENTION_
The use of attention mechanisms in current deep learning
architectures enables models to selectively pay attention
to different parts of the input data; this has enabled the
development of the transformer architecture [11] which is
built solely on multi-head self-attention. The transformer
architecture has changed how we process natural language,
and the success of the transformer architecture has also been
extended to other domains including computer vision.
In the field of computer vision, there are several ways
that attention has been implemented to create more effective



17856 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


models for image classification: spatial attention identifies the
most important parts of the image; channel attention identifies
the most important feature maps; and self-attention identifies
patterns of dependence among all elements of an image or
scene.
At the same time, researchers have begun investigating
quantum-inspired attention mechanisms (QIAMs). QIAMs
apply ideas from quantum physics to the problem of
finding attention weights. Specifically, QIAMs attempt to
find inspiration in the idea of the relationship between
the probability of measuring a particular property of a
physical system, and the way that one finds the probability of
measuring a particular property. In terms of finding attention
weights, features are treated as quantum states, and attention
weights are found through application of Born’s rule. Through
assignment of complex-valued amplitudes to features, and
using the normalized squared magnitudes of those amplitudes
to calculate attention weights, QIAMs can take advantage
of interference effects (i.e., interactions between features),
that classical attention mechanisms cannot. Researchers have
demonstrated the ability of QIAMs to produce better results
than classical attention mechanisms across a variety of tasks
including natural language processing, image classification,
and recommendation systems.
Our work builds on previous research in two ways. First,
we develop a quantum-inspired attention mechanism that
uses learnable complex-valued amplitudes to assign values
to the feature embeddings that are produced by a CNN
(Convolutional Neural Network) backbone. Then, we used
quantum probabilities derived from the squared magnitudes
of normalized amplitudes to determine the importance of each
feature embedding. We then scale and combine the output
of the attention mechanism to produce a representation of
the input image, and we use scaled-dot product attention to
identify relationships between features in the representation.
Finally, we include residual connections to help maintain
stable gradients during training. Second, we develop a method
of selecting features based on QAOA (Quantum Approximate
Optimization Algorithm), and we incorporate the selected
features into the representation of the input image to produce
a final representation of the input image. We believe that
our approach will be able to leverage the advantages of both
quantum and classical approaches to produce more accurate
results for deepfake detection than either classical approach
alone.


_D._ _RESEARCH GAP AND POSITIONING_
There have been many studies on classical methods for
detecting deepfakes. Though quantum computing can be
used to improve machine learning, there are only few studies
applying quantum-inspired techniques to detect deepfakes.
Therefore, in an attempt to fill the lacuna, we present here a
hybrid model. It combines the use of established CNN-based
architectures, as well as quantum inspired optimization and
attention mechanisms. It will serve as a practical solution
to improve the quality of deepfake detection. In contrast



to previous work, which included quantum principles in
general machine learning processes. The proposed system was
created to take advantage of specific attributes of deepfake
detection, such as; the high dimensional nature of facial
features, subtle manipulation artifacts that may appear when
manipulating video or images, the class imbalance associated
with deepfakes (i.e., most data is of non-deepfakes), and
the need for forensic interpretation of results from deepfake
detection. The inclusion of quantum-inspired approaches
to multimedia forensics expands the capabilities of both
deepfake detection and quantum machine learning; therefore,
it represents a new area of research for each community.


**III.** **QUANTUM-HYBRID DEEPFAKE DETECTION**
**FRAMEWORK: SYSTEM OVERVIEW**
The purpose of this portion is to illustrate the organization of
the entire Quantum-Hybrid Deepfake Detection Framework’s
system components by showing how the classical and
quantum inspired components function as one overall
system component in an end-to-end process to process
the input video data; utilizing a combined pipeline of
classical features extraction, quantum inspired optimization
and attention mechanisms to produce output that will be used
for authenticating the video data for use in forensic analyses.


_A._ _SYSTEM ARCHITECTURE AND PROCESSING PIPELINE_
The Deepfake Detection System is comprised of 7 stages
of processing: Data Curation, Preprocessing, Classical
Feature Extraction, Quantum-Inspired Feature Selection,
Quantum-Inspired Attention Weighting, Classification, and
Interpretability Analysis. The entire system architecture is
depicted in Figure 1, which shows how data flows through both
the classical component and the quantum inspired components
of the framework. It composed of an overall pipeline that
starts with raw video, goes through pre-processing, follows
classical feature extraction via InceptionResnetV1, and then
enters into the quantum inspired modules. A QAOA (Quantum
Approximate Optimization Algorithm) module uses the
most discriminative features to determine the best set of
features to use for classification. Then, the re-weighting of
the selected features in the QAOA module occurs using
a quantum inspired attention mechanism to prepare for
final classification. Upon completion of the classification
process, there will be a prediction as well as a number of
different visualizations for understanding how the model
reached its prediction such as Grad-CAM (Gradient Class
Activation Map).
The system architecture is an assembly of 7 modules that
are connected and contribute important functions to the endto-end detection pipeline.


1) INPUT LAYER AND DATA CURATION
The system ingests raw video data from the FaceForensics++
dataset, which contains both authentic videos and manipulated
videos produced using multiple deepfake generation techniques (Face2Face, FaceSwap, DeepFakes, NeuralTextures).



VOLUME 14, 2026 17857


**FIGURE 1.** The architecture of the Quantum-Hybrid deepFake detection
framework.


This module handles format conversion, resolution
standardization, and an initial assessment of video quality.
Metadata such as manipulation type, compression level,
and ground-truth authenticity labels are recorded to
support systematic evaluation across diverse manipulation
strategies.


2) PREPROCESSING MODULE
Frames are sampled every second to create a varied yet nonredundant video dataset.



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


MTCNN is employed to detect and isolate facial areas
within each frame. The cropped facial images are normalized
to maximize the representation of facial features and minimize
the presence of background artifacts.
Blurred images are filtered out using the Laplacian variancebased blur detection metric. The filtered-out images produce a
clearer and more discriminatory training dataset than a dataset
containing low-quality images.
Training robustness and invariance to various manipulations
is enhanced through the application of data augmentation
techniques including random flip, rotation, color jitter, and
affine transformation.


3) CLASSICAL FEATURE EXTRACTION BACKBONE
InceptionResNet-V1 is the classical feature extraction backbone utilized in the proposed method. The InceptionResNet-V1
architecture is pre-trained on the large scale VGGFace2
dataset. Multiple scales of convolution and residual
connections enable the InceptionResNet-V1 architecture
to extract rich 512 dimensional facial feature embeddings.
The facial feature embeddings contain facial identification,
expression, texture, and geometric characteristics. The
extracted facial feature embeddings are the input to the
proposed quantum inspired modules.


4) QUANTUM-INSPIRED OPTIMIZATION LAYER (QAOA)
The Quantum-Inspired Optimization Layer (QAOA) is
employed to select the most informative features from the
facial feature embeddings to address the ‘‘curse of dimensionality’’ problem of the deep CNN feature embeddings.
The QAOA formulates the feature selection problem as a
combinatorial optimization problem. A cost Hamiltonian is
formulated using:

  - Feature importance (calculated using mutual information
between class labels and features) and

  - Correlation penalties (to prevent selecting redundant
features).
A mixer Hamiltonian is defined to allow the quantuminspired exploration of the feature subset space. COBYLA
is used to update the circuit parameters to minimize the
expected energy of the Hamiltonian. The minimization
process of the Hamiltonian results in the convergence to an
optimal 256 dimensional feature subset. The optimal feature
subset improves the computational efficiency of the network,
reduces overfitting, and retains the most discriminative
information.


5) QUANTUM-INSPIRED ATTENTION MODULE
The selected features are then processed by the QuantumInspired Attention Mechanism. Each feature is associated with
a complex number value

_ψi_ = _aie_ _[i][θ]_


where both _ai_ and _θi_ are trainable parameters. The quantum
probability values | _ψi_ | [2] are calculated after normalizing the
complex number values based on Born’s rule. The quantum



17858 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


probabilities indicate how much each feature contributes to
the overall feature representation.
Multi-head attention with 8 heads is performed on the
quantum modulated features. The output of all 8 heads are
concatenated and linearly transformed. Residual connections
and layer normalization are implemented to ensure the stability
of the gradient flow during backpropagation.


6) CLASSIFICATION HEAD
The Classification Head is designed to convert the attention
weighted feature representations to binary classification
decisions. The Classification Head consists of the following
components:

  - Global Average Pooling to reduce the dimensionality of
the feature representation,

  - Dropout Regularization (dropout rate of 30%) to prevent
overfitting,

  - Fully Connected Layers to calculate the class logit values,
and

  - Softmax Activation Function to compute the probability
values for each class label.
The class label corresponding to the maximum predicted
probability is considered to be the predicted class.


7) OUTPUT AND INTERPRETABILITY LAYER
Additional outputs besides the classification results include:


_a:_ _GRAD-CAM HEATMAP_
Spatial maps indicating which facial regions have the
greatest influence on the classification decisions. The heatmap
provides insight into whether the model is relying on facial
expressions, geometry, etc. when making its classification
decisions.


_b:_ _QUANTUM ANALYSIS OUTPUTS_
Additional visualization outputs to provide insight into the
quantum-inspired decision-making process. The additional
outputs include the quantum probability values, feature
importance distribution, and attention patterns.


_c:_ _INTERACTIVE INTERFACE_
An interactive Gradio-based interface allows users to upload
their own images and evaluate the model in real time. Users
can upload an image and receive immediate predictions along
with visual explanations.


_B._ _OPTIMIZATION TRAINING FRAMEWORK_
The training framework uses an optimised version of the
loss function, the focal loss formulation, with three different
components to handle the unbalance of the classes and to
increase the calibration of the model.


1) WEIGHTING OF FOCAL
The focal term ( _γ_ = 2) reduces the weights of correctly
classified sample so that the model focuses more on difficult
samples to classify during the training phase.



2) CLASS BALANCING
The class-balancing factor ( _α_ = 0 _._ 75) has the effect
of increasing the importance of minority-class samples in
comparison with majority-class samples, so that the learning
of the model is still possible even if there is a class imbalance
in the data set.


3) SMOOTHING LABEL
The target distribution is smoothed using label smoothing
( _ε_ = 0 _._ 10), so that it is reduced the over-confidence of the
model and also its robustness to learn decision boundaries.


4) PENALTY CONFIDENCE
A penalty term ( _β_ = 0 _._ 30) is used to penalise large differences
between the confidence that the model assigns to the correct
class and the actual correctness of the classification of the
sample, so that the model is more calibrated.
Finally, the use of the AdamW Optimizer with a weight
decay of 0 _._ 01, in order to have adaptive learning rates and
effective L2 Regularisation, and the use of a OneCycleLR
Scheduler that cyclically varies the learning rate in order to
accelerate the convergence of the model and to maintain the
stability of the training phase. Finally, the integration with
weights & biases, that allows to monitor in real time all the
metrics of training, the trend of the losses, the learning rate
schedules, the utilisation of the resources of the systems during
the entire training phase.


_C._ _INTEGRATION AND WORKFLOW SUMMARY_
The Quantum-Hybrid Framework is a hybrid architecture
where both classical and quantum-inspired modules are
integrated to provide a complete deepfake detection system.


1) CLASSICAL CNN COMPONENTS
take advantage of pre-trained architectures based on large
scale face recognition models (as example FaceNet) to obtain
robust hierarchical feature representations of the facial images.
These pre-trained architectures provide strong generalization
priors and are the base of the quantum-inspired modules.


2) QUANTUM-INSPIRED OPTIMIZATION (QAOA)
enables the efficient combinatorial search of high dimensional
feature spaces in order to identify the discriminative feature
subsets that heuristic methods of classical cannot identify.
In this way, the system exploits the enhanced exploration of
the solution space of the problem through the formulation of
the selection of features as a problem of quantum-inspired
optimization.


3) QUANTUM-INSPIRED ATTENTION MECHANISM
utilizes the principles of quantum interference to model
complex relationships between features. Therefore, the
framework is able to identify the subtle manipulation
signatures of the facial image through the assignment of complex amplitudes to the features and the



VOLUME 14, 2026 17859


derivation of probabilities of quantum that modulate their
importance.


4) BALANCED FOCAL LOSS
ensures that the model learns robustly under significant class
imbalance while maintaining calibrated probability estimates.
It is an essential requirement for deployment in forensic
contexts.


5) FRAMEWORKS OF INTERPRETABILITY
provide spatial, feature-level and quantum-level explanations
to enhance transparency and reliability. These visual and
analytical outputs bridge the gap between closed box
predictions of deep learning and standards forensic that
demand traceable reasoning.
In total, the above components constitute a cohesive
quantum-hybrid system that represents a new paradigm
for deepfake detection. The framework demonstrates that
quantum-inspired principles can meaningfully enhance classical deep-learning pipelines even when implemented using
conventional hardware. The following sections will present
the mathematical formulations, implementation details, and
experimental validation of each component.


**IV.** **METHODOLOGY: DETAILED IMPLEMENTATION**
Our methodology is designed to be reproducible and easily
evaluable by others. Therefore, we have clearly defined all our
methodologies so they may be easily duplicated and rigorously
evaluated.


_A._ _PIPELINE DATA CURATION AND PREPROCESSING_
1) DATASET PROPERTIES AND SAMPLE PLAN
The FaceForensics++ dataset [1] is composed of 1000 unmanipulated YouTube videos and 4000 manipulated videos made
with 4 different deepfake creation methods:

1) Face2Face (the face re-enacts expressions from the
source video while retaining those expressions).
2) FaceSwap (exchanging identities between the source
and target).
3) DeepFakes (GAN-based face reconstruction), and
4) NeuralTextures (expression transfer through neural
texture rendering).


Each video has 150-300 frames at 30 FPS, providing
significant temporal variety among manipulation types.
Frames were sampled at 1 FPS to provide 200-300 frames per
video. This sampling plan meets three goals:

1) Sufficient temporal variation to capture a variety of
facial expressions and head positions.
2) Keep the dataset large enough to train but small enough
to process; and
3) Reduce redundant sampling where there are many
similar consecutive frames.

After cleaning and pre-processing, the curated dataset
consists of around 500,000 high-quality facial images.



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


2) MTCNN BASED PIPELINE FOR FACIAL DETECTION
We used the Multi-Task Cascaded Convolutional Network
(MTCNN) [8] for face detection. It is a cascade of 3 stages
designed to accurately detect faces:
**Stage 1 - Proposal Network (P-Net):** A fully convolutional
network scans an image using a sliding 12 × 12 windows of
multiple sizes based on image pyramids. For each window,
P-Net provides a face or non-face classification score and
bounding box coordinates. P-Net produces coarse candidate
face regions.
**Stage 2 - Refine Network (R-Net):** Candidate regions are
resized to 24 × 24 and then run through R-Net. R-Net refines
bounding box predictions and eliminates false positives with
a more selective classification stage. R-Net produces refined
coordinates and classification scores.
**Stage 3 - Output Network (O-Net):** Remaining candidates
are resized to 48 × 48 for processing through O-Net. O-Net
performs the following tasks:
1) High-accuracy binary face classification.
2) Bounding box regression for accurate location of the
bounding box.
3) Detects five points on the face (eyes, nose, corners of
mouth) for geometric consistency checks.


_a:_ _PARAMETER CONFIGURATION_
We configured MTCNN to use:
1) select_largest = False to allow for detecting
multiple faces.
2) post_process = False to output raw tensors
compatible with PyTorch.
3) Confidence thresholds of [0.6, 0.7, 0.7] for P-Net, R-Net
and O-Net respectively to achieve a balance of precision
and recall.
Cropped detected faces with 30% padding to maintain
context within the region of interest and resized to 160 ×
160 pixels to fit the input size requirements for the
InceptionResnetV1 backbone. The cropped faces are saved in
a hierarchical directory structure:

dataset/{train,val,test}/{real,fake}/

This allows us to easily separate data into the appropriate
subsets and manipulation classes for further training and
evaluation.


3) LAPLACIAN VARIANCE BLUR DETECTION
Sharpness of an image is assessed using the Laplacian operator,
a second order derivative filter that detects the presence of
edges in an image:




_[∂]_ [2] _[I]_ _[∂]_ [2] _[I]_

[+]
_∂x_ [2] _∂y_ [2]



∇ [2] _I_ ( _x, y_ ) = _[∂]_ [2] _[I]_



(1)
_∂y_ [2]



The variance of Laplacian responses quantifies edge
sharpness:



_V_ Laplacian = [1]

_N_



_N_



_i_ =1




- �2
∇ [2] _Ii_ - _µ_ ∇2 (2)



17860 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


The term _N_ represents the number of all pixels in an image;
the second part of this equation is the Laplacian of pixel _i_ (i.e.,
∇ [2] _Ii_ ); and _µ_ ∇2 represents the average of all the values of the
Laplacian for each pixel. High variance results when sharp
images have many large, abrupt edges, which produce many
abrupt changes in intensity. On the other hand, when images
are blurry, there will be fewer abrupt changes in intensity (due
to gradual changes), therefore lower variance in the image.
To implement the Laplacian, we used OpenCV’s laplacian
function with a 3 × 3 kernel to measure the second-order
spatial derivatives at each pixel of the image. All images that
have _Vlaplacian_ _<_ 100 are removed from the data set because
they were classified as blurry. To determine the threshold of
100, we visually inspected samples of images with varying
degrees of blur, and chose a threshold that would remove the
largest possible percentage of the most blurry images without
removing too many clear images. The removal of these blurry
images improved the quality of our data set by about 8−12%.


4) BROAD DATA AUGMENTATION
During training-time augmentation, the following six data
augmentations are used to improve the robustness of the
model:
1) **Random horizontally flipped images (probability** =
**50%):** This will add left right view invariant images.
2) **Random** **rotation** **of** **images** **(** ± **15** **degrees):** The
random rotation will allow the model to be trained on a
variety of head pose variations and camera angles.
3) **Color Jitter:** Brightness, Contrast, Saturation, and Hue
(factor = 20%) will be randomly jittered independently
to provide a variety of lighting conditions and camera
sensors.
4) **Random Affine Transformation:** The affine transformation includes translation (±10%), scale (90–110%),
and shear (±10%). These random transformations will
create a wide variety of Perspective Transformations.
5) **Random** **Gaussian** **Blur** **(Kernel** **Size:** 3 × 3 **,**
**probability** = **20%):** The random blur will increase
the model’s robustness against compressions artifacts.
6) **Normalization:** All pixel values will be normalized to
have an average value of zero and standard deviation of
one using ImageNet statistics.
Each epoch during training, the augmentation is applied
randomly, effectively increasing the diversity of the data set.


_B._ _CLASSICAL BASELINE ARCHITECTURE:_
_InceptionResnetV1_
The traditional baseline uses InceptionResnetV1 [3],
a hybrid architecture that combines two major architectural
innovations:


1) INCEPTION MODULES FOR THE CAPTURE OF FEATURES
AT MULTIPLE SCALES
Inception modules extract multi-scale features from input
images by processing them through parallel convolutional
paths (1 × 1, 3 × 3, 5 × 5 convolutions and 3 × 3 max pooling).



where _yc_ is the ground truth one-hot encoded label and _y_ ˆ _c_ is
the predicted probability for class _c_ . The Adam optimizer with
learning rate 1e-4 and weight decay 1e-5 provides adaptive
learning rates and L2 regularization.



This multi-path design is highly effective for deepfake
detection because manipulation artifacts occur at different
granularities, such as pixel-level anomalies in compressed
regions and region-level inconsistencies in blended areas.
Each Inception module merges feature maps from its parallel
branches, allowing the combined output to encode both
fine-scale textures and coarse semantic information. The
1 × 1 convolutions have two roles: they reduce the
dimensionality before the expensive 3×3 and 5×5 operations,
and they facilitate cross-channel information mixing.


2) RESIDUAL CONNECTIONS FOR FLOW OF GRADIENTS
Residual connections [21] add the input of each Inception
module to its output through identity shortcut pathways:


**y** = _F_ ( **x** _,_ { _Wi_ }) + **x** (3)


where **x** is the input, _F_ ( **x** _,_ { _Wi_ }) represents the Inception
module’s transformation with weights { _Wi_ }, and **y** is the output.
The identity shortcut allows for a direct path of gradient
back-propagation, which greatly reduces vanishing gradient
problems in deep networks.


3) PRE-TRAINING AND TRANSFER LEARNING
The backbone is initialized with weights pre-trained on
VGGFace2 [7], a large-scale face recognition dataset
containing 3.31 million images of 9,131 identities. Pre-training
provides robust facial feature representations encoding
identity-discriminative characteristics that prove beneficial
for manipulation detection. Transfer learning substantially
accelerates convergence (reducing training epochs by 40–
50%) and improves final performance by 3-5 percentage points
compared to random initialization.


4) CLASSIFICATION HEAD ARCHITECTURE
Following feature extraction, the classification head
comprises:
1) **Global Average Pooling** : Reduces spatial dimensions
by averaging across all spatial locations for each feature
channel, producing a 512-dimensional feature vector
invariant to input spatial variations.
2) **Dropout Regularization** (rate=0.3): Randomly zeros
30% of activations during training, preventing coadaptation of neurons and improving generalization.
3) **Fully Connected Layers** : Two dense layers (512 →
512 → 2) with ReLU activation map features to binary
class logits.
The model is trained with standard cross-entropy loss:



_L_ CE = −



2


_yc_ log( _y_ ˆ _c_ ) (4)

_c_ =1



VOLUME 14, 2026 17861


It is important to emphasize that this InceptionResNetV1
configuration serves as the classical baseline for all
comparative experiments. The proposed quantum-hybrid
framework reuses the same InceptionResNetV1 backbone
and preprocessing pipeline, but replaces the standard fully
connected classification head with a QAOA-based feature
selection module and a quantum-inspired multi-head attention
mechanism. This design choice allows the contribution of
the quantum-inspired components to be isolated by directly
comparing the classical baseline and the quantum-hybrid
model under identical feature extraction conditions.


_C._ _QUANTUM-HYBRID MODEL ARCHITECTURE_
1) QAOA-BASED FEATURE SELECTION: MATHEMATICAL
FORMULATION
The feature selection problem seeks to identify a binary vector
**z** ∈{0 _,_ 1} [512] where _zi_ = 1 indicates the feature _i_ is selected
and _zi_ = 0 indicates rejection. The optimization objective
balances two competing goals:


_a:_ _MAXIMIZE DISCRIMINATIVE POWER_
Select features with high mutual information _I_ ( _fi_ ; _y_ ) with
respect to class labels _y_, where mutual information quantifies
the reduction in uncertainty about _y_ given the knowledge of
the feature _fi_ .


_b:_ _MINIMIZE REDUNDANCY_
Penalize selection of highly correlated features, as redundant
features increase dimensionality without adding discriminative
information.
The cost function is formulated as:



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


Hamiltonian applies the transverse field:



_HM_ =



512

- _σi_ _[x]_ (7)

_i_ =1



enabling quantum tunneling between solution candidates
through superposition of |0⟩ and |1⟩ states.


_d:_ _QAOA CIRCUIT_
The parameterized quantum circuit alternates between the cost
and mixer evolutions for _p_ = 3 layers:



| _ψ_ ( _**γ**_ _,_ _**β**_ )⟩=



_p_


_e_ [−] _[i][β][j][H][M]_ _e_ [−] _[i][γ][j][H][C]_ |+⟩ [⊗][512] (8)

_j_ =1



2




_cijzizj_ + _µ_
_i<j_




- 512




_zi_   - 256

_i_ =1



_C_ ( **z** ) = −



512





- _wizi_ + _λ_ 
_i_ =1 _i<j_



(5)



where:

  - _wi_ = _I_ ( _fi_ ; _y_ ) represents feature importance scores
computed via mutual information

  - _cij_ = |corr( _fi, fj_ )| denotes absolute Pearson correlation
between features _i_ and _j_

  - _λ_ = 0 _._ 3 is the redundancy penalty weight

  - _µ_ = 0 _._ 5 is the cardinality constraint weight enforcing
selection of exactly 256 features


_c:_ _QUANTUM ENCODING_
The cost function is encoded as a quantum Hamiltonian by
mapping binary variables to Pauli-Z operators:



_cijσi_ _[z][σ]_ _j_ _[ z]_
_i<j_



_HC_ = −



512





- _wiσi_ _[z]_ [+] _[ λ]_ 
_i_ =1 _i<j_



2

    
1 − _σi_ _[z]_ - 256
2



+ _µ_




- 512



_i_ =1



(6)



where | _ψ_ ( _**γ**_ _,_ _**β**_ )⟩ is the quantum state after circuit application,
_**γ**_ = ( _γ_ 1 _, . . ., γp_ ) and _**β**_ = ( _β_ 1 _, . . ., βp_ ) are variational
parameters, and |+⟩ [⊗][512] is the uniform superposition initial
state.


_e:_ _CLASSICAL OPTIMIZATION_
The expected cost ⟨ _HC_ ⟩ = ⟨ _ψ_ ( _**γ**_ _,_ _**β**_ )| _HC_ | _ψ_ ( _**γ**_ _,_ _**β**_ )⟩ is
minimized using COBYLA (Constrained Optimization BY
Linear Approximations), a derivative-free optimizer suitable
for noisy objective functions typical of quantum circuit
simulation. After convergence (typically 150-200 iterations),
the quantum state is measured, and the bitstring with highest
probability indicates the optimal feature subset.


_f:_ _IMPLEMENTATION DETAILS_
QAOA simulations were conducted using Qiskit and the
statevector_simulator back-end because of the 2 [512]

dimensional Hilbert space. To reduce the complexity from
_O_ (2 [512] ), we used a block-wise approximation where we split
our features into 16 blocks of 32 features each and then
applied QAOA to each block individually (and selected the top
16 features from each block). This was done to maintain near
optimal solutions as well as keep the computation tractable.


2) QUANTUM-INSPIRED ATTENTION MECHANISM
The goal of the quantum-inspired attention module is to model
feature importance and interactions using principles analogous
to quantum states and measurement. Intuitively, each feature
embedding produced by the InceptionResNetV1 backbone
is treated as a ‘‘quantum state’’ with an associated complexvalued amplitude. The squared magnitude of this amplitude
represents how important that feature is, in analogy to the
way that Born’s rule computes measurement probabilities
in quantum mechanics. These probabilities are then used to
reweight the features before applying a standard multi-head
self-attention mechanism. In this way, the module can capture
subtle interference-like effects and non-linear relationships
between features, while still remaining compatible with
conventional deep learning toolchains.
The quantum-inspired attention layer assigns a complexvalued amplitude to each of the 256 selected features. Let _fi_



where _σi_ _[z]_ [are] [Pauli-Z] [operators] [acting] [on] [qubit] _[i]_ [,] [with]
eigenvalues +1 (selected) and −1 (rejected). The mixer



17862 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


denote the _i_ -th feature and let

_ψi_ = _aie_ _[j][θ][i]_ _,_ _ai, θi_ ∈ R _,_ _ai_ _>_ 0 _,_


where _ai_ and _θi_ are learnable parameters initialized randomly.
To obtain a valid quantum-inspired probability distribution,
we normalize the amplitudes according to Born’s rule:


| _ai_ | [2]
| _ψi_ | [2] = �256 _[,]_
_j_ =1 [|] _[a][j]_ [|][2]

so that [�] _i_ [|] _[ψ][i]_ [|][2] [=][ 1. These quantum probabilities modulate]
the feature importance, and the modulated features are
computed as


_f_ ˜ _i_ = | _ψi_ | [2]                  - _fi._


Formally, let **f** = [ _f_ 1 _, f_ 2 _, . . ., f_ 256] denote the selected
feature embedding and [˜] **f** = [ _f_ [˜] 1 _,_ _f_ [˜] 2 _, . . .,_ _f_ [˜] 256] be the probabilityweighted embedding. The multi-head attention mechanism
then operates on [˜] **f** in four main steps:
1) _Complex_ _amplitude_ _assignment_ _and_ _probability_ _com-_
_putation:_ Each feature _fi_ is associated with a complex
amplitude _ψi_ = _aie_ _[j][θ][i]_, and the corresponding
probability _pi_ = | _ψi_ | [2] is obtained by normalizing the
squared magnitudes as above.
2) _Feature reweighting:_ The original features are modulated using these probabilities to yield a probabilityweighted feature vector [˜] **f**, where informative dimensions
are emphasized and less relevant ones are attenuated.
3) _Multi-head_ _self-attention:_ The batch of quantummodulated features is denoted by _F_ [˜] ∈ R _[B]_ [×][256] . For each
head _k_ ∈{1 _, . . .,_ 8}, query, key, and value projections
are computed as

_Q_ [(] _[k]_ [)] = _FW_ [˜] _Q_ [(] _[k]_ [)] [+] _[ b]_ [(] _Q_ _[k]_ [)] _[,]_ (9)

_K_ [(] _[k]_ [)] = _FW_ [˜] _K_ [(] _[k]_ [)] + _b_ [(] _K_ _[k]_ [)] _[,]_ (10)

_V_ [(] _[k]_ [)] = _FW_ [˜] _V_ [(] _[k]_ [)] + _b_ [(] _V_ _[k]_ [)] _[,]_ (11)

where _WQ_ [(] _[k]_ [)] _[,][ W]_ _K_ [ (] _[k]_ [)] _[,][ W]_ _V_ [ (] _[k]_ [)] ∈ R [256][×] _[d][k]_ are learnable
projection matrices (with _dk_ = 32 per head). Scaled
dot-product attention for head _k_ is then given by



which helps maintain stable gradients and preserves
information from the original quantum-modulated
representation.
By combining quantum-inspired probabilistic weighting
with multi-head self-attention, the module is able to
highlight discriminative facial features and capture complex
dependencies between them, which is crucial for detecting
subtle deepfake artifacts.


3) BALANCED FOCAL LOSS FUNCTION
We employed a sophisticated loss function, incorporating
multiple components to combat the class imbalance and to
improve calibration:


_L_ Total = _L_ BFL + _L_ LS + _L_ CP (12)


_a:_ _BALANCED FOCAL LOSS [10]_

_L_ BFL = − _αt_ (1 − _pt_ ) _[γ]_ log( _pt_ ) (13)


where _pt_ = _p_ if _y_ = 1 (fake class) and _pt_ = 1− _p_ if _y_ = 0 (real
class), _αt_ = _α_ for fake samples and _αt_ = 1− _α_ for real samples
( _α_ = 0 _._ 75 favoring minority class), and _γ_ = 2 is the focusing
parameter that down-weights well-classified examples.


_b:_ _LABEL SMOOTHING_



2

  
_L_ LS = − _ϵ_


_c_ =1



1
2 [log(] _[y]_ [ˆ] _[c]_ [)] (14)




         - _Q_ ( _k_ )( _K_ ( _k_ ))⊤
ATn [(] _[k]_ [)] ( _Q, K_ _, V_ ) = softmax ~~√~~
_dk_




_V_ [(] _[k]_ [)] _,_



where the scaling factor [√] _dk_ prevents excessively sharp
attention distributions and stabilizes gradients during
training.
4) _Aggregation, residual connection, and normalization:_
The outputs of all heads are concatenated and linearly
projected back to the original dimensionality:



with _ϵ_ = 0 _._ 1, this function replaces hard targets [0,1] with
soft targets [0.05, 0.95] in order to avoid overly confident
predictions.


_c:_ _CONFIDENCE PENALTY_


_L_ CP = _β_  - max(0 _,_ max( _y_ ˆ _c_ ) − ⊮[prediction correct]) (15)


where _β_ = 0 _._ 3. ⊮[·] is the indicator function. This term
explicitly penalizes high confidence on incorrect predictions,
improving calibration.


_D._ _INTERPRETABILITY VIA GRAD-CAM_
Gradient-weighted Class Activation Mapping [9] produces
class-discriminative localization maps by backpropagating
gradients from the predicted class score to the final
convolutional layer. For target class _c_, the importance weight
for feature map _k_ is:



_αk_ _[c]_ [=] _Z_ [1]






_i,j_



_∂y_ _[c]_

(16)
_∂A_ _[k]_ _ij_



MH( _F_ [˜] ) = Concat�ATn [(1)] _, . . .,_ ATn [(8)][�] _WO_ + _bO,_


where _MH_ means multihead, _ATn_ means Attention,
and _WO_ ∈ R [256][×][256] is the output projection matrix.
A residual connection and layer normalization are
applied to form the final output:


Output = LayerNorm� _F_ ˜ + MH( _F_ ˜ )� _,_



where _A_ _[k]_ _ij_ [denotes] [the] [activation] [at] [spatial] [location] [(] _[i][,][ j]_ [)] [in]
feature map _k_, _y_ _[c]_ is the pre-softmax class score, and _Z_ = _H_ ×
_W_ is the spatial size. These weights represent the importance
of each feature map for the target class.
The Grad-CAM heatmap is computed as:



��     
_L_ Grad-CAM _[c]_ [=][ ReLU] _αk_ _[c][A][k]_

_k_



(17)



VOLUME 14, 2026 17863


The ReLU operation retains only positive influences, as we
are interested in features that increase the class score. The
heatmap is bilinearly upsampled to match the input image
resolution and overlaid with transparency for visualization.


1) IMPLEMENTATION
Grad-CAM targets the final convolutional block _block8_
of InceptionResnetV1, which captures high-level semantic features while retaining sufficient spatial resolution
(10×10) for meaningful localization. The implementation uses
PyTorch’s automatic differentiation, registering forward and
backward hooks to capture activations and gradients during
inference.


_E._ _IMPLEMENTATION FRAMEWORK AND TRAINING_
_PROTOCOLS_
1) SOFTWARE AND HARDWARE INFRASTRUCTURE
The framework is implemented using:


  - **PyTorch 2.0** : Deep learning framework with automatic
differentiation

  - **Qiskit 0.43** : Quantum computing framework for QAOA
simulation

  - **Gradio 3.50** : Interactive web interface development

  - **Weights & Biases** : Experiment tracking and visualization

  - **OpenCV 4.8** : Computer vision operations

  - **NumPy, SciPy** : Numerical computing and optimization

Training infrastructure consists of NVIDIA A100 GPUs
(40GB VRAM) with CUDA 11.8, enabling batch sizes up to
32 for the classical baseline and 24 for the quantum-hybrid
model (due to additional memory requirements of complexvalued parameters).


2) TRAINING PROTOCOL
_a:_ _CLASSICAL BASELINE_

  - Epochs: 50 with early stopping (patience=7 based on
validation loss)

  - Batch size: 32

  - Optimizer: Adam (lr=1e-4, betas=(0.9, 0.999), weight
decay=1e-5)

  - Learning rate schedule: ReduceLROnPlateau (factor=0.5,
patience=3)

  - Loss: Cross-entropy


_b:_ _QUANTUM-HYBRID MODEL_

  - Pre-training: QAOA feature selection (2 hours on CPU,
one-time cost)

  - Epochs: 40 with early stopping (patience=5)

  - Batch size: 24

  - Optimizer: AdamW (lr=1e-4, betas=(0.9, 0.999), weight
decay=0.01)

  - Learning rate schedule: OneCycleLR (max_lr=1e-3,
epochs=40, pct_start=0.3)

  - Loss: Balanced focal loss with label smoothing and
confidence penalty



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


OneCycleLR implements a cyclical learning rate policy:
linear warmup from initial to maximum learning rate during
the first 30% of training, followed by cosine annealing to nearzero, accelerating convergence while maintaining stability.


_F._ _TRAINING HYPERPARAMETERS AND_
_REPRODUCIBILITY DETAILS_
For reproducibility, this section summarizes the key hyperparameters and implementation details used for both the classical
baseline and the proposed quantum-hybrid framework. All
models were implemented in PyTorch (version 2.1) and trained
on a single NVIDIA RTX 3090 GPU with 24 GB of memory.
_Optimizer_ _and_ _learning_ _rate_ _schedule:_ We employ the
AdamW optimizer with an initial learning rate of 1 × 10 [−][4],
weight decay of 0 _._ 01, and default _β_ parameters ( _β_ 1 =
0 _._ 9 _, β_ 2 = 0 _._ 999). A OneCycleLR scheduler is used with a
maximum learning rate of 3 × 10 [−][4], warm-up fraction of 10%
of the total training steps, and cosine annealing to zero for the
remaining steps, as described in Section V. This schedule is
applied consistently to all baseline models and the quantumhybrid framework.
_Batching, epochs, and early stopping:_ Training is conducted
with a batch size of 64 frames, for a maximum of 100 epochs.
An early stopping criterion with a patience of 10 epochs
monitors validation accuracy and restores the best-performing
checkpoint. All reported results are averaged over three
independent runs with different random seeds to reduce
variance.
_Loss_ _configuration:_ The balanced focal loss combines
four components: focal weighting (focusing parameter _γ_ =
2 _._ 0), positive-class emphasis ( _α_ = 0 _._ 75), label smoothing
(smoothing factor _ϵ_ = 0 _._ 10), and a confidence calibration
penalty with weight _λ_ conf = 0 _._ 30. These hyperparameters
were selected by grid search on the validation split, optimizing
for a combination of F1-score and ECE. The same loss
configuration is used for all models to avoid biasing
comparisons.
_Dataset splits and seeds:_ The FaceForensics++ HQ dataset
is partitioned into 70%/15%/15% train/validation/test splits
at the video level, stratified by class to maintain the real/fake
ratio across splits. For all experiments, we fix random seeds
for Python, NumPy, and PyTorch to 42 to ensure deterministic
data shuffling and weight initialization as far as the underlying
libraries allow.
_Code and configuration availability:_ The full training and
evaluation code, including configuration files specifying all
hyperparameters and dataset paths, are made available in an
accompanying repository, enabling external researchers to
reproduce the reported results and adapt the framework to
other datasets.


**V.** **EXPERIMENTAL SETUP AND EVALUATION METRICS**
_A._ _DATASET PARTITIONING_
The faceforensics ++ processed data-set was split into
training (70%), validation (15%), and testing (15%) splits,



17864 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


**TABLE 1.** Comparative performance evaluation on the FaceForensics **++** HQ benchmark.



stratified as to maintain class distribution balance through-out
each split. performance on the test set will be used to determine
final unbiased results for the trained models. the validation set
will be used to help identify optimal values of hyperparameters
and guide early stopping of model development.


_B._ _EVALUATION METRICS_
Comprehensive evaluation employs six complementary
metrics:

  - **Accuracy:** Overall correctness, ( _TP_ + _TN_ ) _/_ ( _TP_ + _TN_ +
_FP_ + _FN_ )

  - **Precision:** Positive predictive value, _TP/_ ( _TP_ + _FP_ ),
critical for minimizing false accusations.

  - **Recall:** True positive rate, _TP/_ ( _TP_ + _FN_ ), ensuring
detection of actual deepfakes.

  - **F1-Score:** Harmonic mean of precision and recall, 2 ·
( _P_    - _R_ ) _/_ ( _P_ + _R_ ).

  - **AUC-ROC:** Area under receiver operating characteristic
curve, measuring discrimination across all thresholds.

  - **ECE:** Expected calibration error, quantifying alignment
between predicted confidence and actual correctness:



as the proposed quantum-hybrid framework but employs
a standard fully connected classification head without
QAOA-based feature selection or quantum-inspired
attention. This configuration isolates the contribution
of the quantum-inspired components.

  - **MesoNet (Classical)** : A compact CNN designed specifically for facial forgery detection, optimized for local
texture artifacts.

  - **Capsule-Forensics** **(Classical)** : A capsule networkbased approach that models part–whole relationships
and has been shown to be effective for detecting forged
images and videos.
_Training protocol consistency:_ All baseline methods and
the proposed quantum-hybrid framework share the following
protocol:

  - Training data: FaceForensics++ HQ split (1000 authentic and 4000 manipulated videos spanning Face2Face,
FaceSwap, DeepFakes, and NeuralTextures), using the
same frame sampling strategy described in Section IV-A.

  - Preprocessing: Identical MTCNN-based face detection,
160 × 160 resizing, Laplacian variance-based blur filtering, and the same data augmentation pipeline (horizontal
flip, rotation, color jitter, affine transformations, and
Gaussian blur).

  - Data partitioning: Stratified train/validation/test splits of
70%/15%/15% by video, preserving class balance across
all splits.

  - Training schedule: Maximum of 100 epochs with early
stopping based on validation accuracy (patience of
10 epochs).

  - Optimization: AdamW optimizer with weight decay 0 _._ 01,
and learning rate scheduling as described in Section IV-E,
applied consistently across all models.

  - Loss: Balanced focal loss with label smoothing and
confidence penalty for all methods to avoid biasing
results due to different loss formulations.
For methods without pre-trained deepfake-specific weights,
the original architectural specifications were reimplemented
in PyTorch and trained end-to-end under the protocol
above. This setup ensures that the comparison between the
proposed quantum-hybrid framework and the baselines is both
methodologically consistent and empirically fair.


**VI.** **RESULTS AND ANALYSIS**
_A._ _QUANTITATIVE PERFORMANCE COMPARISON_
Table 1 presents a comprehensive comparison of the proposed
quantum-hybrid deepfake detection framework against five



ECE =



_M_



_m_ =1



| _Bm_ |

_N_ [|][acc(] _[B][m]_ [)][ −] [conf(] _[B][m]_ [)][|] (18)



where _M_ = 10 bins partition predictions by confidence, _Bm_ is
the set of samples in bin _m_, acc( _Bm_ ) is accuracy within bin _m_,
and conf( _Bm_ ) is average confidence in bin _m_ .


_C._ _BASELINE METHODS AND COMPARATIVE EVALUATION_
_PROTOCOL_
To ensure a comprehensive and fair evaluation, multiple stateof-the-art deepfake detection methods were implemented and
evaluated alongside the proposed quantum-hybrid framework.
All baseline methods were trained and evaluated using the
identical FaceForensics++ HQ protocol to guarantee fair
comparison. _Baseline_ _selection:_ The baselines represent
diverse architectural paradigms used in deepfake detection:

  - **Xception** **(Classical)** : A depthwise separable convolutional architecture widely adopted in deepfake
detection due to its efficiency and strong performance on
manipulation artifacts.

  - **EfficientNet-B4 (Classical)** : A family of scaled CNNs
that balance depth, width, and resolution to achieve high
accuracy with improved parameter efficiency.

  - **InceptionResNetV1 (Baseline)** : The classical baseline
in this study. It uses the same feature extraction backbone



VOLUME 14, 2026 17865


state-of-the-art classical methods on the FaceForensics++
HQ benchmark. All methods are trained and evaluated using
identical preprocessing pipelines, data splits (70%/15%/15%),
and evaluation metrics. The proposed quantum-hybrid
framework consistently outperforms strong classical baselines
across accuracy, precision, recall, F1-score, AUC-ROC, and
calibration (ECE). Lower ECE is better; all other metrics are
higher-is-better.
From an overall detection perspective, the proposed
quantum-hybrid framework achieves an accuracy of 98.5%,
improving upon the strongest classical baseline, CapsuleForensics (96.3%), by 2.2 percentage points and upon the
InceptionResNetV1 baseline (95.8%) by 2.7 percentage
points. This gain is notable because InceptionResNetV1
shares the same backbone as the proposed model; thus, the
observed improvement can be attributed primarily to the
QAOA-based feature selection and quantum-inspired attention
components rather than to differences in the convolutional
feature extractor. The proposed method also outperforms
widely used architectures such as Xception and EfficientNetB4 by margins of 4.3 and 3.4 percentage points in accuracy,
respectively.
The quantum-hybrid model maintains a favourable balance
between precision and recall, attaining 98.2% precision
and 98.7% recall. Compared to Capsule-Forensics (95.8%
precision, 96.7% recall), this corresponds to gains of 2.4 and
2.0 percentage points, indicating that the proposed framework
is simultaneously better at avoiding false alarms and at
detecting manipulated content. Such balanced improvements
are particularly important in forensic applications, where
both missed detections and false accusations carry significant
consequences. In terms of F1-score, which jointly summarizes
precision and recall, the quantum-hybrid framework reaches
98.4%, compared to 96.2% for Capsule-Forensics and 95.7%
for the InceptionResNetV1 baseline, confirming that the
improvements are not driven by a single metric.
The proposed method also achieves the highest AUCROC (99.2%), improving on Capsule-Forensics (98.1%)
by 1.1 percentage points and on the InceptionResNetV1
baseline (97.9%) by 1.3 percentage points, demonstrating
superior discrimination across all operating thresholds. This
discrimination capability is consistent with the ROC analysis
shown in Figure 2. The quantum-hybrid model achieves a
higher Area Under the Curve (AUC), indicating superior
discrimination capability across all classification thresholds.
The quantum-hybrid ROC curve dominates that of the
classical baseline over almost the entire range of false-positive
rates.
Calibration, quantified by Expected Calibration Error
(ECE), is particularly critical in forensic and legal contexts.
The quantum-hybrid framework attains an ECE of 0.019,
substantially lower than the ECE values of all classical
baselines (0.068 for Xception, 0.054 for EfficientNet-B4,
0.048 for InceptionResNetV1, 0.082 for MesoNet, and
0.042 for Capsule-Forensics). In relative terms, the calibration
of the proposed model is approximately 2.2× better than



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


**FIGURE 2.** Comparison of receiver operating characteristic (ROC) curves for
the classical baseline and quantum-hybrid models.


that of Capsule-Forensics and about 2.5× better than the
InceptionResNetV1 baseline, indicating that its predicted
probabilities are well aligned with actual correctness. This
behaviour is visually confirmed by the reliability diagram
in Figure 3, where the quantum-hybrid curve lies closer
to the diagonal (perfect calibration). The quantum-hybrid
model’s curve is consistent with its lower Expected Calibration Error (ECE) and improved reliability of confidence
predictions.


**FIGURE 3.** Reliability diagram (calibration plot) comparing the classical
baseline and quantum-hybrid models.


_B._ _ABLATION STUDY: COMPONENT CONTRIBUTIONS_
Table 2 presents ablation study results quantifying individual
component contributions to overall performance.



17866 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


**TABLE 2.** Ablation study: Component contribution analysis.


The results of the ablation study show that each part of
this approach is contributing positively to performance gains.
In addition to a better accuracy and F1 score for feature
selection alone, 1.1 percent and 1.3 respectively, it also shows
the potential of reducing dimensionality intelligently with
QAOA. A second 0.7 percent accuracy and a second 1.3 F1
improvement was made using quantum-inspired attention to
model subtle relationships between features. Finally, another
0.7 percent in both accuracy and F1, while also dramatically
improving calibration (the ECE decreased from 0.058 to
0.045), were achieved through balanced focal loss. The
overall results demonstrate the advantage of combining
multiple quantum-inspired components as they produce
positive synergies together.


_C._ _CROSS-MANIPULATION GENERALIZATION_
The hybrid Quantum Model is also tested in experiments
to determine if it has the ability to generalize over different
types of video manipulation (manipulations) as opposed to
simply being able to identify specific manipulations used
during training. For example, both models were trained on
three datasets; Face2Face, FaceSwap and Deepfakes, which
are examples of some of the most commonly used video
manipulation tools. After the models were trained on these
three datasets, they were then tested using the NeuralTextures
dataset that was withheld from all previous training (the
test set). The results of testing the two models using the
NeuralTextures dataset indicated that the Hybrid Quantum
Model had an accuracy rate of 89.2%, while the Classical
Baseline model had an accuracy rate of 85.7%. Therefore,
there is a 3.5% difference between the performance of the
Hybrid Quantum Model and the Classical Baseline model
when testing on a manipulation type that neither model
was trained on. As shown in Figure 4, this increase in
generalizability may be attributed to the Hybrid Quantum
Models’ ability to identify common manipulation-agnostic
discriminative features, as opposed to manipulating techniquespecific artifacts. Figure 4 shows the performance comparison
across individual deepfake manipulation techniques from the
FaceForensics++ dataset, including generalization to the
unseen NeuralTextures technique. The quantum-hybrid model
consistently outperforms the classical baseline, demonstrating
robust and generalizable detection capabilities.


_D._ _COMPUTATIONAL EFFICIENCY ANALYSIS_
The increased computational time of the hybrid-quantum
model to perform its additional steps is offset by the
considerable dimensionality-reduction from applying the



**FIGURE 4.** Accuracy comparison across individual deepfake manipulation
techniques.


QAOA-based features selected for each sample to the quantumhybrid model; however, it requires slightly longer than the
classical model for the same number of samples as the
classical model takes about 0.8 sec to complete per batch,
whereas the quantum-hybrid model takes approximately
1.1 sec to complete per batch or an increase of 37.5%. The
additional processing time is considered reasonable for the
large performance increases and increased interpretability
that are obtained with the quantum-hybrid model. The
QAOA feature selection process (which is performed
only one-time during initial model development) required
approximately 2 hr on standard CPU hardware using the blockwise approximation strategy, providing a proof-of-practical
feasibility for deploying the model into real-world applications.
The model’s training history is shown in Figure 5. The plots
show accuracy and loss curves over training epochs. The
quantum-hybrid model demonstrates faster convergence and
achieves a lower validation loss, indicating more efficient and
effective learning.


**FIGURE 5.** Training and validation history for the classical baseline and
quantum-hybrid models.


_E._ _INTERPRETABILITY ANALYSIS_
Distinctive patterns of attention occur among the models
as illustrated by the Grad-CAM visualization. The classical
baseline model shows a dispersed pattern of attention across
most of the facial area, whereas the quantum-hybrid model
focuses attention much more clearly on the features relevant
to forensic analysis such as eyes (28% of attention) and mouth



VOLUME 14, 2026 17867


borders (22%) and facial contours (18%), and nose (15%).
Such focused attention may be indicative that the quantum
inspired features of the model, which feature importance and
attention weights are shown in Figures 6 and 7, allow the
model to recognize subtle manipulations which are typically
present in artifacts of deepfakes due to challenges with
synthesizing expressions and inconsistencies from blending.
The plot 6 highlights that QAOA prioritizes a subset of highly
discriminative features while discarding redundant or less
informative ones, leading to a more efficient and focused
feature representation.


**FIGURE 6.** Distribution of feature importance scores for the 256 features
selected by the QAOA module.


**FIGURE 7.** Visualization of quantum-inspired attention weights assigned to
the selected features.


Figure 7 data distribution shows that the model learns to
dynamically assign higher importance to specific features,
effectively modeling complex inter-feature relationships and
interference patterns. Quantum-hybrid model attention (seen
in Figure 8) is much closer to where the manipulation artifacts
are located as described by forensic investigators than the
attention produced by the classical models. It shows a sample
image, the heatmap from the classical model, and the heatmap
from our quantum-hybrid model. The quantum-hybrid model
shows more focused attention on the eyes and mouth, which
are common areas for manipulation artifacts.
The interactive tool provided for the user (in Figures 9 and 10). It includes all mentioned analysis tools. They
will allow forensic investigators to review new evidence
quickly and efficiently in ‘‘real-time’’ to produce actionable,
interpretable reports from their analysis.
The Quantum-Hybrid CNN model’s interface (figure 9)
displays the classification result, detected face, attention map,



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


**FIGURE 8.** Grad-CAM visualization comparison between classical baseline
and quantum-hybrid models.


and detailed metrics. The detailed metrics include confidence
distribution, layer-wise analysis, and feature importance,
providing a comprehensive and interpretable analysis for the
end-user.
Similar to the user interface of the quantum-hybrid model,
the Classical Inception-ResNet model (figure 10) provides a
comprehensive analytical toolset that enables users to directly
compare the model’s output with its attention and all of the
performance metrics used to evaluate it. This side-by-side
analysis of the model’s output, attention and performance
metrics is critical to understanding the value of the quantum
inspired components of the model.


**FIGURE 9.** User interface of the quantum-hybrid CNN model.


**VII.** **DISCUSSION**
_A._ _QUANTUM-INSPIRED SOLUTIONS BENEFITS_
This research demonstrates the potential for quantum inspired
algorithms (QIAs) to offer a genuine benefit in the detection
of deepfakes using classical Hardware. By employing feature
selection via the QAOA, the researchers were able to mitigate
the curse of dimensionality inherent in the representation of
deep convolutional neural networks (CNNs). And therefore
employ combinatorial optimization principles inspired by



17868 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


**FIGURE 10.** User interface of the classical Inception-ResNet model.


quantum computing to identify optimal subsets of features.
The quantum inspired attention mechanism employed by the
researchers enabled the modeling of interference patterns,
enabling the identification of subtle inter-feature relationships
that would likely remain unidentifiable to classical attention
mechanisms.
In addition to the increasing body of evidence demonstrating
that Quantum Inspired Classical Algorithms will enhance
machine learning performance prior to the advent of fully
fault-tolerant quantum computers; the current research
offers additional validation of the utility of hybrid systems
of quantum-classical systems employing actual quantum
hardware. The 2.5% increase in accuracy and 45% reduction
in error covariance ellipse due to the employment of quantum
inspired components lends credence to the notion that
this paradigm has practical utility in multimedia forensics
applications.


_B._ _FORENSIC APPLICATIONS AND DEPLOYMENT_
_CONSIDERATIONS_
The Grad-CAM visualizations and quantum analysis metrics
are particularly beneficial in forensic applications because
they enable better understanding of the technology. In legal
and investigative contexts, not only is accurate classification
required, but also transparency into how the conclusions were
drawn so that the results can be evaluated and validated by
qualified forensic domain experts. The hybrid model’s ability
to focus on facial regions most relevant to forensic application,
as well as to quantify the relative importance of individual
features in the model. It provides a means for forensic domain
experts to evaluate the results of the model and to develop
trust in the results produced by the model. The model’s use
of quantum probability distributions also provides a natural
interface for understanding the contribution of each feature to



each prediction, thereby closing the gap between the internal
workings of the model and the human interpretation necessary
for forensic applications.
Like the improvement in interpretability, the enhanced
calibration (ECE=0.045) will also facilitate deployment of the
model. Users of models that are well-calibrated can rely upon
the confidence scores provided by the model to inform their
decision-making with regard to classification; and this could
lead to practitioners evaluating uncertain cases (confidence
50−70%), while allowing them to automatically classify
highly confident classifications (confidence - 90%). This
‘‘human-in-the-loop’’ approach enables investigators to utilize
automation efficiently while maintaining investigative rigor,
and ultimately to achieve the highest level of throughput
possible while adhering to the same high standards of forensic
quality.


_C._ _LIMITATIONS AND FUTURE DIRECTIONS_
Some limitations are worthy of acknowledgement and suggest
possible avenues for future study:


1) HARDWARE VERSUS SIMULATION
Classical simulation of a quantum algorithm is used; whereas,
a near-term, simulated quantum device has limited potential
to improve performance using genuine superposition and
entanglement. Although, some near-term quantum computers
have relatively high gate error rates (0.1 - 1%) and low
coherence times (microsecond to millisecond) which create
challenges in implementing practical machine learning
applications.


2) GENERALIZING RESULTS
FaceForensics++ dataset represents controlled laboratory
conditions that include specific types of manipulations,
although there are many types of manipulated images in
social media, journalism and forensic contexts that may
exhibit very different artifact distributions, compression
levels and image quality. It is essential to evaluate the
robustness of this method over several real world datasets
(e.g., DFDC, Celeb-DF). Preliminary results on Celeb-DF
indicate an accuracy of 87.3% for the quantum-hybrid
model compared to 84.1% for the classical baseline.
These preliminary results suggest some level of real-world
applicability.


3) INCREASED INFERENCE TIME
An increase in inference time of 37.5% is considered
acceptable for a forensic batch processing application,
however it can be detrimental to real-time applications
deployed in resource constrained edge environments. Future
studies should examine techniques for compressing models
(e.g., pruning, quantization), knowledge distillation from
quantum-hybrid teacher models to efficient student models,
and optimization of quantum circuit simulations using tensor
network methods to reduce complexity.



VOLUME 14, 2026 17869


4) ROBUSTNESS TO ADVERSARIAL ATTACKS
This framework’s ability to detect adversarial attacks that
were specifically developed to avoid detection needs to be
extensively evaluated. Future studies need to evaluate the
performance of this framework under adversarial attacks
(FGSM, PGD attacks) and develop strategies for adversarial
training that incorporate principles for generating robustness
inspired by quantum computing.


5) TEMPORAL ANALYSIS
The experiments in this paper focus on detecting deepfakes at
the frame level. Adding temporal analysis to identify temporal
inconsistencies in deepfakes using 3-D CNNs, temporal
attention, LSTM aggregation, etc. could lead to improved
detection accuracy by 3−5% based on previous studies.


_D._ _BROADER IMPACT AND ETHICAL CONSIDERATIONS_
The detection of deepfakes has been an important step
in the protection of individual privacy and as a defense
mechanism against the spread of misinformation. However, the
development of such technologies has introduced many ethical
considerations. Deepfake detection technology can be misused
through censorship, surveillance, and/or discrimination
against certain demographic or political views. As the
continuous cat-and-mouse game develops with increasing
sophisticated generations of deepfakes and improved detection
technologies, there is likely to be a false sense of security
resulting in a greater difficulty in detecting generated content.
For the deployment of responsible technology, transparency
in both the communication of detection capability, as well
as limitations, is necessary. In addition, the potential impact
of false positives on legitimate content creators must be
considered. Ongoing evaluations of bias in the application
of this technology to different demographic groups are
important. Our initial assessment of our detection tool has
demonstrated consistent accuracy across age demographics
(accuracy variation ± 1.2%) and gender (accuracy variation
± 0.8%), however, we feel additional research is needed to
evaluate the fairness of this tool across various ethnicities, skin
tones and socioeconomic statuses.
We support a collaboration of technologists, policymakers,
legal experts, civil society organizations and impacted
communities to develop governance structures which balance
the need for security requirements with protections for civil
liberties. Technical solutions alone cannot address the social
implications of deepfakes; therefore, media literacy programs,
accountability mechanisms from platforms, and appropriate
legislation that penalizes maliciously created and disseminated
deepfakes while providing protection for legitimate creative
works and scientific research are essential.


_E._ _INTEGRATION WITH EXISTING SYSTEMS_
Quantum hybrid is intended for easy integration in current
video forensics workflows. QAOA can be used with the current
CNN based forensic tools to increase their performance. The



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


quantum inspired attention can be added to already trained
CNN’s via fine tuning. The interpretability tool will provide all
users with a standardized output format which will allow them
to generate reports using forensic reporting tools. Gradio is
an example of how you can build a user friendly UI (User
Interface) that hides all of the complex technical details
from the forensic analyst so they can use advanced detection
techniques that are otherwise out of their reach. API end
points also exist so that a forensic lab can run batches of
videos through the detection software at once to speed up
investigations. Webhooks are also available so that when new
content is posted to Facebook or YouTube it can automatically
be sent to your forensic lab to be processed.


**VIII.** **CONCLUSION**
In this paper, we present a complete hybrid quantum
framework for the detection of deepfakes using convolutional
neural networks (CNN) and quantum inspired optimization
and attention mechanisms. We use QAOA based feature
selection, quantum inspired attention modeling and balanced
focal loss optimization to address three main limitation in
classical deepfake detection systems; feature redundancy, class
imbalance and limited interpretability.
Our experimental results on the FaceForensics++ benchmark, show that the proposed quantum-hybrid architecture
achieves higher performance in all the metrics used for
evaluation. Specifically, we observe significant improvements
in terms of precision (+3.4%), AUC-ROC (+0.022) and
calibration Error (−45%) compared to classical baseline
models. The ablation studies performed to evaluate the
individual contribution of each of the quantum inspired
components confirms their meaningful impact on the
improvement of the overall performance, confirming the
synergy of the integration strategy proposed. Moreover, our
cross-manipulation generalization experiments indicate that
the 3.5% points advantage of the quantum-hybrid model
compared to the classical baselines extends to unknown
deepfake techniques and therefore, it is able to learn
manipulation-agnostic features.
The Grad-CAM Visualization shows that the quantumhybrid model develops more focused attention on the most
relevant regions for forensic analysis of the face, i.e., eyes,
mouth boundary and facial contours, where artifacts due
to manipulation are usually found. Therefore, the quantumhybrid model provides better interpretability for expert review.
Additionally, the integration of the quantum analysis metrics
such as the distributions of the feature importance, the quantum
probability patterns, and the visualizations of the weights of
the attention, provide an unprecedented level of transparency
into the processes of decision making of the models. And it
bridges the gap between the closed box predictions and the
forensic requirements.
Finally, this work provides a solid basis for future quantumenhanced multimedia forensic systems. The architectural
principles and hybrid paradigms presented in this paper,
such as the quantum inspired combinatorial optimization



17870 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


for feature selection, complex valued attention modeling
interference patterns and multi-objective loss functions
balancing accuracy and calibration. It represents transferable
methodologies applicable to diverse forensic challenges such
as audio deepfake detection, synthetic text identification, and
multimedia tampering localization.
Quantum computing technologies are currently maturing
and becoming more accessible and when quantum hardware
becomes available, the simulation-based QAOA implementation will be possible to migrate to real quantum processors
and thus, unlock potential performance improvements through
true quantum superposition and entanglement.
Future near-term quantum devices with 100 to 1000 qubits
could make possible to explore larger feature spaces
without block-wise decomposition. And future error-corrected
quantum computers could extend quantum-hybrid approaches
to more complex optimization landscapes including joint
architecture and hyperparameters search.


_A._ _SUMMARY OF CONTRIBUTIONS_
Our key contributions are,
1) **New Hybrid-Quantum Architecture** : Complete deepfake detection system that combines QAOA feature
selection with quantum inspired multi-head attention
to show how quantum principles can enhance classical
deep learning even when implemented on conventional
hardware.
2) **Theoretical** **Framework** **for** **DeepFake** **Detection** :
Complete theory of a formalized mathematically-based
framework of feature selection using quantum Hamiltonian optimization, complex valued attention amplitudes
defined by Born’s rule probability computation, and
balanced focal loss with calibration penalty.
3) **Full** **Scale** **Experiments** : Quantitative experimental results for improved accuracy (2.5%), reduced
calibration error (45%) and generalization across
manipulations, plus detailed ablation studies quantifying
contributions of each component.
4) **New Interpretable Framework** : Incorporating GradCAM spatial visualizations with quantum analysis
metrics providing multi-level interpretations from
spatial heat maps of attention to feature level importance
quantifications and quantum probability distributions.
5) **Practical Implementation** : An open source framework
with interactive Gradio interface, complete documentation and modular design allowing forensic practitioners
and researchers to adopt components incrementally.
6) **Methodology** **Advances** : Showing that quantuminspired approaches can address specific machine
learning challenges like dimensionality reduction,
complex relationships between features and even before
fault-tolerant quantum computers are available.


_B._ _BIGGER PICTURE_
Quantum-inspired methodologies for deep-fake detection have
the opportunity to augment state-of-the-art methodologies for



deep-fake detection. Improved methodologies for deep-fake
detection would allow forensic labs, content providers and
investigative agencies to more easily verify authenticity of
digital evidence. Also, since our approach produces enhanced
interpretability of the models used for detection, expert
witnesses will be able to provide transparent explanations
about how they reached conclusions and thus be compliant
with legal requirements for admissibility of the witness’s
testimony.
In a larger sense, the use of quantum-inspired algorithms
for multimedia forensics provides a good example of how
ideas from quantum computing can be applied to support
impactful applications in real-world scenarios. If successful,
these applications can stimulate other researchers to look
into additional applications of the principles of quantum
computing.
Finally, the open-source nature of our detection methodology (as well as its modular design) was intended to encourage
researchers and practitioners to develop the methodology
further, and thus foster a collaborative research community
working to address the deep-fake problem.
Our long-term vision is that this work is a major step
towards creating more robust information systems against
many forms of malicious behavior, including manipulation or
creation of digital content. In general, this should help maintain
journalistic integrity, due process in courts, democratic
disclourse and protect individuals’ private information as
technologies which generate synthetic digital content continue
to grow and evolve. Since we believe that the use of quantuminspired innovations will continue to offer viable options for
detecting manipulated digital content, our proposed detection
paradigm is a potentially important contribution to maintaining
trust in digital media.


_C._ _CALL TO ACTION_
We would like to see the research community use our
foundation to move in a number of different ways: including
video-level temporal modeling using hybrid-quantum techniques; evaluating true implementations on actual quantum
hardware when it matures; adding capabilities to enhance
adversarial robustness; and performing a wide-ranging
evaluation of fairness across various demographics. In order to
unlock the full potential of the quantum-enhanced multimedia
forensics paradigm while concurrently addressing the ethical
considerations, collaboration will be required among quantum
computing researchers, computer vision specialists, forensic
scientists, and ethicists.
To enable the reproduction of results, and allow
practitioners to apply the quantum-hybrid framework in
their particular forensic context, all of the code, trained
models, and an interactive demonstration are located
here: https://github.com/fakubwoy/QuantDeepfakeDetection.
We welcome community contributions and collaborative
research to advance the state-of-the-art in both deepfake
detection and quantum-inspired machine learning.



VOLUME 14, 2026 17871


In summary, we believe that the quantum-hybrid framework
represents a major advancement in applying quantuminspired computing paradigms to multimedia forensics. The
framework is based on measurable improvements in detection
accuracy, calibration, interpretability, and generalization,
compared to the prior art. Our work demonstrates the
viability and practicality of applying quantum-inspired
optimization and attention mechanisms to classical deep
learning systems. This provides the research community with
a number of new avenues to investigate the intersection
of quantum computing and digital forensics. We believe
that this work inspires continued innovation in quantumenhanced multimedia security, leading to greater levels of
trustworthy and resilient digital information ecosystems
that benefit society while preserving individual rights and
freedoms.


**ACKNOWLEDGMENT**
The authors wish to thank the computational resources
and infrastructure which were provided by Vellore Institute
of Technology, specifically GPU clusters for performing
extensive experimental validations, and high performance
computing resources for simulating Quantum Approximate
Optimization Algorithm (QAOA) [4], [5]. The authors would
also like to extend special gratitude to the creators of the
FaceForensics++ dataset [1] for providing this data set to
be used as a benchmark; it has allowed for the systematic
evaluation and fair comparisons to existing state of the
art methods of deepfake detection systems. The authors
would like to express gratitude to the developers of PyTorch,
Qiskit, Gradio, and Weights and Biases, for developing
high-quality open source software tools that have greatly
facilitated rapid prototyping, thorough experimentation, and
clear visualization of results. The authors would like to
extend appreciation to members of the Computer Vision
and Machine Learning Research Group at Vellore Institute
of Technology, Vellore for engaging discussions, feedback
on initial results, and assistance with curation of datasets.
A special thanks goes out to the students that volunteered
in preliminary user studies assessing the usability of the
Gradio interface and provided insight into requirements for
practitioners in terms of interpretability. Finally, the authors
would like to acknowledge the broader research community
working on deepfake detection, quantum machine learning,
and multimedia forensics, for their contributions to theory and
empirical findings that formed the basis for the approach taken
in this research.


**REFERENCES**


[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Rieß, J. Thies, and
M. Nießner, ‘‘FaceForensics++: Learning to detect manipulated facial
images,’’ in _Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)_, Jan. 2019,
pp. 1–11.

[2] W. Yu, P. Zhou, S. Yan, and X. Wang, ‘‘InceptionNeXt: When
inception meets ConvNeXt,’’ in _Proc._ _IEEE/CVF_ _Conf._ _Comput._
_Vis._ _Pattern_ _Recognit._ _(CVPR)_, Seattle, WA, USA, Jun. 2024,
pp. 5672–5683.



F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


[3] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, ‘‘Inception-v4,
Inception-ResNet and the impact of residual connections on learning,’’
in _Proc. AAAI Conf. Artif. Intell._, vol. 31, San Francisco, CA, USA, 2017,
pp. 4278–4284.

[4] E. Farhi, J. Goldstone, and S. Gutmann, ‘‘A quantum approximate
optimization algorithm,’’ 2014, _arXiv:1411.4028_ .

[5] R. Shaydulin, ‘‘Evidence of scaling advantage for the quantum approximate
optimization algorithm on a classically intractable problem,’’ _Sci._ _Adv._,
vol. 10, no. 22, pp. 1–10, May 2024.

[6] V. M. Rathod, A. M. Patil, H. S. Motekar, M. Usmani, V. D. Solavande, and
S. B. Rathod, ‘‘Automatic face recognition based on enhanced VGGFace16 model in an unconstrained environment using transfer learning,’’
_Multimedia Tools Appl._, vol. 84, no. 33, pp. 41741–41763, Apr. 2025, doi:
[10.1007/s11042-025-20819-w.](http://dx.doi.org/10.1007/s11042-025-20819-w)

[7] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, ‘‘VGGFace2:
A dataset for recognising faces across pose and age,’’ in _Proc._ _13th_
_IEEE_ _Int._ _Conf._ _Autom._ _Face_ _Gesture_ _Recognit._ _(FG)_, May 2018,
pp. 67–74.

[8] S. Karamizadeh, S. Shojae Chaeikar, and H. Salarian, ‘‘Combining
MTCNN and enhanced FaceNet with adaptive feature fusion for robust
face recognition,’’ _Technologies_, vol. 13, no. 10, p. 450, Oct. 2025, doi:
[10.3390/technologies13100450.](http://dx.doi.org/10.3390/technologies13100450)

[9] M. A. I. Aminudin, M. N. Abdullah, F. Mustapha, K. K. Eng, M. Mustapha,
and A. Mustapha, ‘‘Explainable deep learning framework for binary
corrosion image classification using grad-CAM,’’ _Sensors_, vol. 25, no. 22,
[p. 7070, Nov. 2025, doi: 10.3390/s25227070.](http://dx.doi.org/10.3390/s25227070)

[10] F. Z. E. Biach, I. Iala, H. Laanaya, and K. Minaoui, ‘‘Efficient balanced
focal loss function for manipulated images detection,’’ in _Proc._ _5th_
_Int._ _Conf._ _Intell._ _Comput._ _Data_ _Sci._ _(ICDS)_, Oct. 2021, pp. 1–4, doi:
[10.1109/ICDS53782.2021.9626750.](http://dx.doi.org/10.1109/ICDS53782.2021.9626750)

[11] Q. Wang, L. Wang, M. Fu, J. Wang, L. Sun, R. Huang, X. Li, Z. Jiang, and
H. Luo, ‘‘Multiscale transformer and attention mechanism for magnetic
spatiotemporal sequence localization,’’ _IEEE Internet Things J._, vol. 11,
no. 11, pp. 19454–19469, Jun. 2024, doi: [10.1109/JIOT.2024.3365793.](http://dx.doi.org/10.1109/JIOT.2024.3365793)

[Online]. Available: https://ieeexplore.ieee.org/document/10436406/

[12] P. Senapati, S. Y.-C. Chen, B. Fang, T. M. Athawale, A. Li, W. Jiang,
C. C. Lu, and Q. Guan, ‘‘PQML: Enabling the predictive reproducibility
on NISQ machines for quantum ML applications,’’ in _Proc._ _IEEE_ _Int._
_Conf._ _Quantum_ _Comput._ _Eng._ _(QCE)_, Sep. 2024, pp. 1413–1424, doi:
[10.1109/QCE60285.2024.00168.](http://dx.doi.org/10.1109/QCE60285.2024.00168)

[13] E. K. Mounika, S. K. Shareef, M. Adudhodla, N. K. Sripada, D. Karuru, and
M. Bhavsingh, ‘‘Quantum feature pruning for scalable and efficient quantum
kernel-based high-dimensional classification,’’ in _Proc. Int. Conf. Inventive_
_Comput. Technol. (ICICT)_, Kirtipur, Nepal, Apr. 2025, pp. 1412–1420, doi:
[10.1109/ICICT64420.2025.11005048.](http://dx.doi.org/10.1109/ICICT64420.2025.11005048)

[14] I. Amerini, L. Galteri, R. Caldelli, and A. Del Bimbo, ‘‘Deepfake video
detection through optical flow based CNN,’’ in _Proc._ _IEEE/CVF_ _Int._
_Conf. Comput. Vis. Workshop (ICCVW)_, Seoul, South Korea, Oct. 2019,
pp. 1–10.

[15] E. M. S. Reddy, A. P. Kumar, and P. Swetha, ‘‘Deepfake video detection
using CNN and RNN with OPTICAL FLOW features,’’ in _Proc._ _IEEE_
_Int. Students’ Conf. Electr., Electron. Comput. Sci. (SCEECS)_, Feb. 2024,
[pp. 1–7, doi: 10.1109/SCEECS61402.2024.10482344.](http://dx.doi.org/10.1109/SCEECS61402.2024.10482344)

[16] G. Petmezas, V. Vanian, K. Konstantoudakis, E. E. I. Almaloglou, and
D. Zarpalas, ‘‘Video deepfake detection using a hybrid CNN-LSTMtransformer model for identity verification,’’ _Multimedia Tools Appl._, vol. 84,
[no. 33, pp. 40617–40636, Mar. 2025, doi: 10.1007/s11042-024-20548-6.](http://dx.doi.org/10.1007/s11042-024-20548-6)

[17] X. Yang, Y. Li, and S. Lyu, ‘‘Exposing deep fakes using inconsistent head
poses,’’ in _Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)_,
May 2019, pp. 8261–8265.

[18] W. Ahmad, Y.-T. Peng, and Y.-H. Chang, ‘‘FAME: A lightweight
spatio-temporal network for model attribution of face-swap deepfakes,’’ _Expert_ _Syst._ _Appl._, vol. 292, Nov. 2025, Art. no. 128571, doi:
[10.1016/j.eswa.2025.128571.](http://dx.doi.org/10.1016/j.eswa.2025.128571)

[19] H. H. Nguyen, J. Yamagishi, and I. Echizen, ‘‘Capsule-forensics:
Using capsule networks to detect forged images and videos,’’ in _Proc._
_IEEE_ _Int._ _Conf._ _Acoust.,_ _Speech_ _Signal_ _Process._ _(ICASSP)_, May 2019,
pp. 2307–2311.

[20] M. Tan and Q. V. Le, ‘‘EfficientNet: Rethinking model scaling for
convolutional neural networks,’’ in _Proc._ _36th_ _Int._ _Conf._ _Mach._ _Learn._
_(ICML)_, Long Beach, CA, USA, 2019, pp. 6105–6114.



17872 VOLUME 14, 2026


F. Khan et al.: Quantum-Hybrid Framework for Enhanced Deepfake Detection


[21] I. C. Duta, L. Liu, F. Zhu, and L. Shao, ‘‘Improved residual networks for image and video recognition,’’ in _Proc._ _25th_ _Int._ _Conf._
_Pattern_ _Recognit._ _(ICPR)_, Milan, Italy, Jan. 2021, pp. 9415–9422, doi:
[10.1109/ICPR48806.2021.9412193.](http://dx.doi.org/10.1109/ICPR48806.2021.9412193)

[22] R. M. Devadas and T. Sowmya, ‘‘Quantum machine learning: A comprehensive review of integrating AI with quantum computing for computational
advancements,’’ _MethodsX_, vol. 14, Jun. 2025, Art. no. 103318, doi:
[10.1016/j.mex.2025.103318.](http://dx.doi.org/10.1016/j.mex.2025.103318)


FARHAAN KHAN is currently pursuing the B.Tech.
degree in computer science and engineering with
Vellore Institute of Technology, Vellore, India.
He is particularly interested in hybrid
quantum–classical approaches for artificial intelligence (AI) applications and in exploring the
intersection of quantum information theory and
deep learning systems. He has been involved in
research-oriented projects focused on emerging
technologies in artificial intelligence and has
contributed to academic work related to quantum-inspired optimization
techniques. His research interests include quantum computing applications in
machine learning, multimedia forensics, and computer vision.


ADITYA SAREEN is currently pursuing the B.Tech.
degree in computer science and engineering with
Vellore Institute of Technology, Vellore, India.
He has worked on projects involving data
preprocessing, exploratory data analysis, predictive
modeling, and decision-support systems. He has
also been involved in interdisciplinary work that
integrates machine learning with natural language
processing, computer vision, and data-driven
optimization approaches. His current focus is on
developing transparent, scalable, and interpretable data-centric AI solutions to
support informed decision-making. His research interests include data science,
data analytics, and explainable artificial intelligence, with an emphasis on
extracting actionable insights from complex datasets.



AKASH SURESH KUMAR is currently pursuing
the B.Tech. degree in computer science and
engineering with Vellore Institute of Technology,
Vellore, India.
He has worked on various projects related to
model robustness, data augmentation Strategies,
and developing efficient training methodologies for
resource-constrained environments. He has actively
engaged in competitive programming and machine
learning competitions, developing strong practical
skills in implementing state-of-the-art algorithms. He is particularly interested
in the computational efficiency aspects of machine learning systems and
exploring novel approaches to enhance model performance while minimizing
resource requirements. His contributions include algorithm optimization,
distributed training strategies, and efficient inference techniques. His research
interests are deep learning optimization, feature engineering, and adversarial
machine learning.


M. BHUVANESWARI received the M.Tech.
degree in computer science and engineering from
Indian Institute of Technology Madras, Chennai,
Tamil Nadu, India, and the Doctoral degree in
information and communication engineering from
Anna University, Chennai. She is currently an
Associate Professor with the School of Computer
Science and Engineering, Vellore Institute of
Technology, Vellore, Tamil Nadu. Her research
interests include mobile ad-hoc networks, network
security, cybersecurity, and intelligent decision support systems.



VOLUME 14, 2026 17873


