**A Study on Deepfake Detection Methods**


Shivnarayan Ahirwar [1*], Alpana Pandey [1 ]

1 _Department of Electronics and Communication Engineering, Maulana Azad National Institute of Technology, Bhopal,_

_Madhya Pradesh, India_

         - Corresponding author E-mail shivnarayan.ahirwar@gmail.com (Shivnarayan Ahirwar)
**Abstract**


**The continued development of deepfake Generative Adversarial Networks (GANs) have generated much controversy**
**within the multiple fields including politics, entertainment, and healthcare. Deepfakes cause the generation of highly**
**convincing fake news to spread with the aim of influencing public opinions, and given that the pictures are near realistic**
**and the sound quality is excellent, it gets hard to differentiate between original and fake content. The release of apps**
**for creating deepfakes has greatly increased the need to find ways to mitigate them because the consequences are**
**criminal on the levels of fake news and privacy infringement. For instance, deepfake technology is capable of**
**manipulating political messages, deceiving the electorate, and inaccurately portraying diseases with likelyhoods of**
**wrong diagnosis and subsequent wrong treatment. This present paper presents a comprehensive survey and analysis**
**of the existing deepfake detection methods including the traditional machine learning techniques, state of the art deep**
**learning methods, and the recent approach based on blockchain technology. Unlike some previous literature reviews**
**that may only include new literature from the last few years, this research synthesizes results from more recent findings**
**to explicate system limitations and recent developments. The presented research is designed to provide methodological**
**approaches and findings to the discussion on deepfake detection to facilitate further empirical work and practical**
**developments beneficial for the crucial field. For deepfake technology lies on the path of digital media’s continuous**
**growth, therefore critical detection methods must be enhanced and united to maintain the desired information’s purity**
**and shield society against potential risks.**



**Keywords- Deepfake, CNN, GAN, Image Forensics.**
**Introduction**


Over the last few years, deepfake technology has
become a cause for major concern in many
industries and concentrations such as political,
entertainment, and healthcare industries. Deepfakes
using second-generation GAN to generate realistic
fake media can deceive the audiences and control the
perception [1]. Despite the mirco-forensic chances
that the utilized deep learning tools have enabled a
burgeoning of deepfakes, it has become difficult to
differentiate between real and fake contents [2-3].
Informing, misinforming and the privacy violation
aspect of deep fake technology is worrisome to have
around. For instance, deep fake videos have been
deployed to spread fictions that could help shape a
society’s perception of certain political
personalities. Further, in the medical domain,
unscrupulous use of deepfake images of patients
aggravates possible dangers such as wrong diagnose
and mistreatment of the patients [4]. Therefore, there
is the need for strong detection systems to be put in
place to avoid such risks while protecting
information [5].


The purpose of this paper is to present an overview
of the state of the art of deepfake detection
techniques, evaluating the issues encountered by
existing frameworks and the progress that has been
made in recent years. The following outlines various
detection techniques: traditional ML, DL and new
methods such as blockchain[6].



Thus, based on synthesising findings derived from
the studied body of literature accruing in the recent
past, this research aims to enrich the existing
discussion on deepfake detection and may help
propose a suite of directions for future research and
potential practical consideration in this important
area of study [7]. As the terrain of digital media
changes, the need to detect these miscreants
becomes more critical and this is where more
research efforts, technologies and policies needs to
be urgently developed.


**Literature Review**


The detection of deepfake videos has been
occasioned by the fast-growing technologies and an
increased worry of fake media. This literature
review synthesizes findings from various studies,
categorizing them into four primary approaches:
Deep learning learning based methods, traditional
AI approaches, statistical algorithms and solutions
based on blockchain.


1. Artificial Neural Network Based Techniques


Deep learning has proved to be the best in the current
methods of deepfake detection since neural
networks can recognize deepfake distortions. Many
papers have been published that show that CNNs
work well when it comes to identifying deepfakes.
For example, Zhang et al. proposed a GAN
simulator that imitates artifacts created by deepfake
production techniques and uses classifiers to
increase their identification effectiveness. Other



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


models include, XceptionNet as well as ResNet that
performs excellent in the classification of real
images from fake images [8].


A literature synthesis that included 112 papers
published from 2018 through 2020 showed that deep
learning techniques surpassed traditional
approaches, with recognition rates exceeding 89%.
Temporal features including eye blinking and head
movements have also been used and models like
LSTM networks have been used to look at frames
sequences.


2. Classical Statistical Learning Algorithms


, but methods of classical machine learning are still
important, especially when the amount of data is
restricted. Based on the extracted features, the
current work employed Support Vector Machines
(SVM) and Random Forests for classifying deepfake
content. These methods generally depends on low
level features which are manually extracted and
includes face shape, face texture etc to detect
inconsistencies in the forged images.


In any case, the accuracy rate of classical methods is
lower compared to deep learning methods with the
average accuracy not exceeding 80%. This
limitation makes it necessary for future works to
explore better feature extraction methods, and to
incorporate the strategies of deep learning [9].


3. Statistical Techniques


Statistical methods have also been employed to
assess the authenticity of media. These approaches
typically involve analyzing the distribution of pixel
values and other statistical properties to detect
anomalies indicative of manipulation. While these
methods can provide valuable insights, they often
lack the robustness and adaptability of machine



learning techniques, particularly in the face of
evolving deepfake generation methods [10].


4. Blockchain-Based Solutions


New studies have looked into the suitability of
blockchain in combating deepfake and
authenticating content. In other words, the use of
blockchain can bring efficiency into the process of
identifying the true source of digital content. This
seems highly suitable for journalism and legal use,
as the correctness of information is essential in these
fields. Nevertheless, there are still a number of
challenges that take the simple proposal of a
blockchain solution and apply it to deepfake
detection by comparing the chain to the work done
in real life.


From the present literature, the consistent signs point
to the utilization of deep learning techniques in
deepfake detection because of their efficiency and
malleability. Although still prominent, it is clear that
problems with classical machine learning and
statistical approaches explain why the detection
methodologies still require further research and
development. Also, the study of blockchain as a
concept offers an opportunity to strengthen the idea
of digital media credibility. Depending on the
advancements in deepfake technology, a significant
and effective method of prevention and detection of
such images will also make-up a necessity [11].


**A Review on State-of-the-art Deepfake Detection**
**Methods**


Due to a surge in deepfake, the need to come up with
proper detection techniques has arisen. This
comparative study synthesizes findings from various
research papers, categorizing the detection
techniques into four primary approaches given in
Table 1



Table 1. Overview of different deepfake detection methods












|.S.No.|Method|Overview|Performance|Strengths|Weaknesses|
|---|---|---|---|---|---|
|1|Deep<br>Learning-<br>Based<br>Methods|Deep learning techniques,<br>particularly Convolutional<br>Neural Networks (CNNs),<br>have<br>become<br>the<br>cornerstone of deepfake<br>detection. These methods<br>leverage large datasets to<br>learn intricate patterns and<br>features that distinguish<br>real<br>from<br>manipulated<br>media.<br>|Studies<br>indicate<br>that deep learning<br>models<br>achieve<br>accuracy<br>rates<br>exceeding 89% in<br>detecting<br>deepfakes.<br>For<br>instance, the use of<br>architectures like<br>XceptionNet and<br>MesoInception-4<br>has<br>shown<br>promising results.|These methods<br>can<br>automatically<br>extract features<br>from<br>data,<br>making<br>them<br>highly effective<br>in<br>identifying<br>subtle artifacts.<br>|Their<br>reliance<br>on<br>feature<br>engineering<br>makes them less<br>adaptable<br>to<br>new types of<br>deepfakes, and<br>they may not<br>perform well on<br>complex<br>datasets.|



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


|2|Machine<br>Learning-<br>Based<br>Methods|Traditional machine<br>learning techniques, such<br>as Support Vector<br>Machines (SVM) and<br>Random Forests, have<br>been employed to detect<br>deepfakes by analyzing<br>specific features extracted<br>from images and videos.|These methods<br>generally achieve<br>lower accuracy<br>rates (around 70-<br>80%) compared to<br>deep learning<br>approaches. They<br>rely heavily on<br>handcrafted<br>features, which<br>can limit their<br>effectiveness.|Machine<br>learning models<br>are often easier<br>to interpret and<br>require less<br>computational<br>power than<br>deep learning<br>models.|Their reliance<br>on feature<br>engineering<br>makes them less<br>adaptable to<br>new types of<br>deepfakes, and<br>they may not<br>perform well on<br>complex<br>datasets.|
|---|---|---|---|---|---|
|3|Statistical<br>Techniques|Statistical methods focus<br>on analyzing the inherent<br>properties of images and<br>videos to detect anomalies<br>indicative<br>of<br>manipulation.|These<br>methods<br>have shown mixed<br>results,<br>often<br>achieving<br>lower<br>accuracy<br>compared<br>to<br>machine learning<br>and deep learning<br>techniques.|Statistical<br>techniques can<br>be effective in<br>specific<br>contexts, such<br>as<br>identifying<br>unique<br>noise<br>patterns<br>in<br>images.<br>|They<br>are<br>generally<br>less<br>robust<br>against<br>sophisticated<br>deepfake<br>generation<br>techniques and<br>may<br>not<br>generalize well<br>across different<br>datasets.|
|4|Blockchain-<br>Based<br>Solutions|Blockchain<br>technology<br>offers<br>a <br>decentralized<br>approach to verifying the<br>authenticity<br>of<br>digital<br>content,<br>providing<br>a <br>potential<br>solution<br>for<br>deepfake detection.<br>|While still in the<br>experimental<br>phase, blockchain<br>methods<br>show<br>promise<br>in<br>tracking<br>the<br>provenance<br>of<br>media.<br>|These solutions<br>can<br>create<br>immutable<br>records<br>of<br>media, making<br>it<br>easier<br>to<br>verify<br>authenticity.|The<br>practical<br>implementation<br>of<br>blockchain<br>for<br>deepfake<br>detection is still<br>underdeveloped,<br>and challenges<br>remain in terms<br>of<br>scalability<br>and integration<br>with<br>existing<br>systems.|



Table 2. Comparative Summary








|Detection Method|Performance|Strengths|Weaknesses|
|---|---|---|---|
|**Deep Learning**|>89%|High accuracy, automatic<br>feature extraction|Requires<br>large<br>datasets,<br>computationally<br>intensive,<br>struggles<br>with<br>zero-day<br>attacks|
|**Machine Learning**|70-80%|Easier<br>to<br>interpret,<br>less<br>computationally demanding|Relies<br>on<br>feature<br>engineering, less adaptable|
|**Statistical Techniques**|Mixed|Effective in specific contexts|Less<br>robust<br>against<br>sophisticated deepfakes|
|**Blockchain Solutions**|Experimental|Immutable<br>records<br>for<br>authenticity verification|Underdeveloped, scalability<br>issues|



Comparing with the prior and recent studies, the
paper notes that, although DL-based approaches
remain the main focus of deepfake research today,
the development of MDFA will require the
integration of machine learning, statistical analysis,
and the use of such technologies as blockchain. The
reason is that each of the mentioned techniques has



certain advantages and disadvantages and that is
why, further research on the use of different methods
for deepfake detection should be aimed at
combining the above approaches to further develop
the capabilities of deepfake detection systems and to
meet the new tasks posed by this technology.



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


This is helpful because it provides a starting point
for which future studies can commence to start to
build on this from where it was left off and expand
on the present work that has been presented for the
detection of deepfakes.


**The Role of Deep Learning for Deepfake**
**Detection**


Deep learning has become the most effective
solution for the detection of deepfakes because it can
learn features by itself from the massive dataset.
This section briefly describes Deep Learning
techniques, their topology, and their operation with
respect to detecting deep fakes.


**A generality regarding deep learning in deepfake**
**detection**


Specifically, Convolutional Neural Network (CNN)
models are intended to work with graphical data
sets. They are particularly good at working input of
images and videos and are thus appropriate for
detecting faked data. The main strength of deep
learning is that it learns the features or hierarchies in
a more complex manner therefore is able to see
deeper artifacts that might denote manipulation [12].



Key Components of Deep Learning Models


1. Input Layer: The model receives raw image or
video data.


2. Convolutional Layers: These layers apply filters
to the input data to extract features. Each filter
detects specific patterns, such as edges or textures.


3. Activation Functions: Functions like ReLU
(Rectified Linear Unit) introduce non-linearity,
allowing the model to learn complex relationships.


4. Pooling Layers: These layers reduce the
dimensionality of the data, retaining only the most
important features while discarding less significant
information.


5. Fully Connected Layers: After several
convolutional and pooling layers, the data is
flattened and passed through fully connected layers
to make predictions.


6. Output Layer: The final layer produces the
classification result, indicating whether the input is
a real or fake image.



















Figure 1. Deep Learning Model for Deepfake Detection


**Machine Learning-Based Methods for Deepfake**
**Detection**


Traditional ML techniques have been used to detect
deepfake, especially where deep learning solutions
cannot be practiced because of lack of data or lack
of computational resources. They tend to use feature
extraction and classification as a way of identifying
manipulated content. This section describes various
machine learning-based methods, the structure of
the mentioned methods, and their operation in deep
fake detection [13].


**A Brief Introduction to Machine Learning in**
**Deepfake Detection**



Figure 2. working flowchart of deep learning model for
deepfake detection



Machine learning methods for deepfake detection
generally involve two main steps: Two main
divisions exist in our work: feature extraction and
classification. Specific objectives of data analysis



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


are to recognize some peculiar features or changes
in data that could be the result of manipulation.
Some of the known algorithms are Support Vector
Machines (SVM), Random Forest or Decision Trees
and among others [14].


**Key Components of Machine Learning Models**


1. Feature Extraction: This step involves identifying
and extracting relevant features from the input data
(images or videos). Features can include:




- Temporal Features: Changes in facial
expressions or movements over time.

- Texture Features: Characteristics related to
the surface quality of the image.

- Facial Landmarks: Points on the face that
represent key features (e.g., eyes, nose,
mouth).













Figure 3. Machine learning model for deepfake detection



2. Feature Selection: After extraction, the most
relevant features are selected to improve the model's
performance and reduce dimensionality.


3. Classification: The selected features are fed into a
machine learning algorithm to classify the input as
either real or fake. The model is trained on labeled
data, where it learns to distinguish between genuine
and manipulated content.


Figure 3 illustrating the architecture of a typical
machine learning model used for deepfake
detection. Based on the experiments, it is figured out
that it is possible to use machine learning-based
methods instead of deep learning methods for
deepfake detection in cases with the lack of data or
computational resources. As in the case of the
previous approaches, these methods generally do not
provide the same level of accuracy as deep learning
ones, however, they allow one to distinguish
between manipulated and non-manipulated content,
if specific features are used. Current studies
conducted in this direction are focused on improving
the stability and flexibility of machine learning
algorithms to prevent the alterations in deepfake
technology from displacing their effectiveness [1516].


**Blockchain-Based Deepfake Detection Method**
**Original Data Identification**

- Media submitted to the network platform is first
processed to extract a unique data fingerprint
(via hashing) combined with the submitter’s
identity.

- A credibility value is computed using deepfake
detection algorithms, assessing the likelihood of
manipulation.




- The fingerprint, identity, and credibility are
embedded in the media as a digital watermark
using techniques like Quantization Index
Modulation (QIM).

- The modified media and its metadata (e.g., hash
values) are stored on the blockchain for future
verification.
**Deep Forgery Data Forensics and Appraisal**

- **Forensic** **Fixation:** Media hashes and
extraction processes are logged on the
blockchain to ensure tamper-proof evidence.

- **Authenticity Analysis:** Media is compared
with stored blockchain records to verify
modifications.

- **Similarity Analysis:** Media is assessed against
existing blockchain data for content similarity,
using deepfake detection models where needed.

- Results are stored on-chain, ensuring
transparent and immutable records.
**Traceability Mechanism**

- If data appears forged, its digital watermark and
fingerprint are extracted for source tracing.

- The **Hamming distance** between suspicious
and existing data identifiers helps determine
relationships, identifying forged data and its
origin.

- If no match is found, the media is treated as new
and processed for initial identification.
This blockchain-based framework addresses
existing challenges in deepfake detection, such as
the inability to track manipulations and provide
judicially recognized evidence. It integrates
blockchain with forensic technologies, offering a
scalable and reliable solution for combating the
misuse of deepfake media [17].

**Conclusion**
In this study we presented various state-of-the-art
methods, we presented basic techniques and



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


discussed different detection models in this work.
We summarize the overall study as follows.
 Mainly deep learning-based methods are used
for deepfake detection.
 In most of the research, CNN based model is
used.
 The most widely used performance metric is
detection accuracy.


**References**


[1] Karaköse, M., Çeçen, M., & Yetiş, H. (n.d.). A
New Approach for Effective Medical Deepfake
Detection in Medical Images.

[2] Korshunov, P., & Marcel, S. (2018). DeepFakes:
a New Threat to Face Recognition? Assessment and
Detection (Version 1). arXiv.
https://doi.org/10.48550/ARXIV.1812.08685

[3] Rana, S., Nobi, M. N., Sung, A. H., Shohel, M.,
& University of Southern Mississippi. University of
Southern Mississippi. (n.d.). Deepfake Detection: A
Systematic Literature Review.

[4] Reiss, T., Cavia, B., & Hoshen, Y. (2023).
Detecting Deepfakes Without Seeing Any (Version
1). arXiv.
https://doi.org/10.48550/ARXIV.2311.01458

[5] M. S. Rana, M. N. Nobi, B. Murali and A. H.
Sung, "Deepfake Detection: A Systematic Literature
Review," in _IEEE Access_, vol. 10, pp. 25494-25513,
2022, doi: 10.1109/ACCESS.2022.3154404

[6] S. Dasgupta, J. Mason, X. Yuan, O. Odeyomi,
and K. Roy, "Enhancing Deepfake Detection using
SE Block Attention with CNN," _2024 International_
_Conference on Artificial Intelligence, Big Data,_
_Computing and Data Communication Systems_
_(icABCD)_, pp. 1-6, 2024.

[7] B. Ghita, I. Kuzminykh, A. Usama, T. Bakhshi,
and J. Marchang, "Deepfake Image Detection Using
Vision Transformer Models," _2024_ _IEEE_
_International_ _Black_ _Sea_ _Conference_ _on_
_Communications and Networking (BlackSeaCom)_,
pp. 332-335, 2024.

[8] V. Rogovoi, V. M. Korzhuk, and O. A. Kokorina,
"Development of a Deepfake Detection Method:
Application of Frequency Analysis and Reduction of
the Image Color Space to Improve Classification
Accuracy," _2024 V International Conference on_
_Neural_ _Networks_ _and_ _Neurotechnologies_
_(NeuroNT)_, pp. 36-39, 2024.

[9] Y. Yang, N. B. Idris, D. Yu, C. Liu, and H. Wu,
"Decentralized Deepfake Task Management
Algorithm Based on Blockchain and Edge
Computing," _IEEE Access_, vol. 12, pp. 8645686469, 2024.

[10] M. S. Rana, M. N. Nobi, and A. Sung,
"DeepDistAL: Deepfake Dataset Distillation using



Active Learning," _2024 IEEE/CVF Conference on_
_Computer_ _Vision_ _and_ _Pattern_ _Recognition_
_Workshops (CVPRW)_, pp. 7723-7730, 2024.

[11] Y. S. Taspinar and I. Cinar, "Distinguishing
Between AI Images and Real Images with Hybrid
Image Classification Methods," _2024_ _13th_
_Mediterranean_ _Conference_ _on_ _Embedded_
_Computing (MECO)_, pp. 1-4, 2024.

[12] S. Antad, V. V. Arthamwar, R. K. Deshmukh, A.
U. Chame, and H. P. Chhangani, "A Hybrid
Approach for Deepfake Detection using CNNRNN," _2024 OPJU International Technology_
_Conference (OTCON) on Smart Computing for_
_Innovation and Advancement in Industry 4.0_, pp. 16, 2024.

[13] M. A. P. Putra, N. W. Utami, I. G. J. E. Putra, N.
Karna, A. Zainudin, and G. A. R. Sampedro,
"Collaborative Decentralized Learning for
Detecting Deepfake Videos in Entertainment," _2024_
_IEEE_ _Gaming,_ _Entertainment,_ _and_ _Media_
_Conference (GEM)_, pp. 1-4, 2024.

[14] S. K. Ahir and O. M. Adedayo, "Multimedia
Forensics: Preserving Video Integrity with
Blockchain," _2024 12th International Symposium on_
_Digital Forensics and Security (ISDFS)_, pp. 1-6,
2024.

[15] R. Singh, K. Ashwini, B. C. Priya, and K. P.
Kumar, "Deepfake Face Extraction and Detection
Using MTCNN-Vision Transformers," _2024 Third_
_International Conference on Distributed Computing_
_and Electrical Circuits and Electronics (ICDCECE)_,
pp. 1-8, 2024.

[16] A. Gadde, G. D. K. Kishore, T. Talari, S. L.
Nunna, R. C. Nannapaneni, and K. M. S. K. Vamsi,
"Detecting Deepfake Images: A Deep Learning
Approach with Streamlit Integration," _2024_
_International Conference on Science Technology_
_Engineering and Management (ICSTEM)_, pp. 1-7,
2024.

[17] M. Priya, J. Murugesan, P. Bhuvaneswari, M.
Rubigha, S. Lalithambikai and B. Mohanraj,
"Preserving Visual Authenticity: Block chainAugmented AI Frameworks for Advanced Digital
Deception Recognition and Mitigation," _2024 5th_
_International Conference on Smart Electronics and_
_Communication (ICOSEC)_, Trichy, India, 2024, pp.
707-713,
doi:10.1109/ICOSEC61587.2024.10722740.



Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:39:19 UTC from IEEE Xplore. Restrictions apply.


