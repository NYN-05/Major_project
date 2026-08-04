Proceedings of the Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS-2025)
IEEE Xplore Part Number: CFP25CB2-ART; ISBN: 979-8-3315-0724-4

# Real-Time Deepfake Video Detection using Machine Learning: A CNN-based Authentication Framework



_1S. Babitha,_
_Assistant Professor, Information Technology,_
_Hindustan Institute of Technology and Science,_

_Chennai._
_Email: babi2289@gmail.com_


3Dhanush Harsha. J.V,
_Information Technology,_
_Hindustan Institute of Technology and Science,_

_Chennai._
_Email: 21134046@student.hindustanuniv.ac.in_


_**Abstract-**_ **Deepfake technology presents significant threats to**
**digital security, misinformation, and privacy. A Convolutional**
**Neural network-based deepfake video detection in real time**
**using optimization techniques to increase the accuracy and**
**computational efficiency is proposed in this paper. The model is**
**trained on the dataset of 15,000 images and uses feature**
**extraction, adaptive loss function, and real-time processing**
**optimization to achieve better performance. The experimental**
**results give competitive results in terms of accuracy, precision,**
**and recall compared to state-of-the-art models like EfficientNet**
**and Xception. The framework is tolerant to adversarial attacks**
**and compression effects. The work done in this direction will be**
**further extended in the future to build transformer-based**
**models for better generalization and adversarial resilience in**
**deepfake detection.**


_**Keywords-**_ _**Deepfake**_ _**Detection,**_ _**Convolutional**_ _**Neural**_
_**Networks (CNNs), Real-Time Video Authentication, Optimization**_
_**Techniques, Adversarial Robustness**_


I. INTRODUCTION

Deepfake technology, powered by artificial intelligence,
has transformed content creation, enabling the synthesis of
hyper-realistic manipulated videos and images. Deepfakes
have beneficial applications, such as virtual reality and
entertainment, but they also completely undermine digital
security and privacy while spreading misinformation [1].
Generative models like Generative Adversarial Networks
(GANs), along with diffusion models, have rapidly evolved
into deep learning methods which, with increasing
sophistication, have become an increasingly difficult task to
distinguish real media from deepfakes [2]. These advanced
manipulations are robust against the traditional detection
methods, therefore, machine learning-based solutions have to
be very robust. By its very nature, the proliferation of
deepfakes creates moral, legal, and security concerns, such as
using them to spread misinformation, conduct identity theft,
and commit fraud [3]. Limitations in conventional detection
techniques result in social media platforms and forensic



2 _Yadavamuthiah. K,_
_Information Technology,_
_Hindustan Institute of Technology and Science,_

_Chennai._
_Email: 21134002@student.hindustanuniv.ac.in_


4 _Hariharan. J,_
_Information Technology,_
_Hindustan Institute of Technology and Science,_

_Chennai._
_Email: 21134005@student.hindustanuniv.ac.in_


experts not being able to control the spread of manipulated
content, and the lack of generalization across different datasets
and manipulation techniques is what makes that difficult.
Because deepfake content is so realistic, it is challenging to
detect deepfakes and needs models that can generalize across
different deepfake creation methods. Traditional CNN-based
architectures can not properly detect such subtle manipulation
as micro expressions, texture bling, motion inconsistencies,
etc., and typical adversarial attacks and compression artifacts
reduce the accuracy further [4]. There are also real-time
applications that require the best trade-off between detection
accuracy and computational efficiency to perform quick
inference without loss in performance. To tackle these
problems, this paper presents a CNN-based authentication
system for real-time deepfake detection. To overcome this, the
framework employs CNNs for feature extraction and
integrates adaptive loss functions to improve detection
precision and optimization techniques to improve efficiency
computation. Additionally, it also attempts to design hybrid
approaches by combining CNNs with transformers and
recurrent networks to gain extra power in detecting deepfakes.
The study also establishes this tradeoff between speed of
detection and accuracy to have real-time applicability and
compares the proposed method with other state-of-the-art
models like Efficient Network, Xception, and ResNet50. This
work is very important since it improves the robustness to
high-resolution manipulation by adversarial attacks, making
for a scalable and effective solution in the field of digital
security [13].


II. REVIEW OF LITERATURE

The study by Sundaram et al. [8] presents an innovative
approach to deepfake detection by leveraging advanced deep
learning models for video authentication. The work is a
demonstration of the effectiveness of deep neural networks in
video content manipulation detection and improves digital
security. The idea is to integrate deep models with real-time
detection mechanisms for more accuracy and faster



979-8-3315-0724-4/25/$31.00 ©2025 IEEE 134

Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:40:28 UTC from IEEE Xplore. Restrictions apply.


Proceedings of the Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS-2025)
IEEE Xplore Part Number: CFP25CB2-ART; ISBN: 979-8-3315-0724-4



computational efficiency. However, there is still not much to
be done to detect highly sophisticated deepfakes that employ
adversarial techniques, with the requirement of time and
resources. The study also highlights the need to keep updating
the model against modulating deepfake techniques.


Lokhande et al. [9] propose an innovative artificial
intelligence-based framework for the cyberattack detection
triggered by deepfake technology and identity theft. The study
uses machine learning algorithms and pattern recognition
techniques to improve the accuracy, as well as the speed, of
cyber attack identification. The other significant innovation is
that it enables the integration of AI-driven anomaly detection
with real-time monitoring to strengthen cybersecurity against
deepfake-related fraud. The main issue is that for the model to
work, it needs a lot of data to train on, and also, the false
positives that it finds detecting sophisticated deepfake attacks
might not be false positives. All of this is reinforced by the
fact that the study outlines the need to continuously refine AI
models to address changing threats.


Anandhasivam et al. [10] propose an innovative hybrid
MobileNet-LSTM model for deepfake detection,
incorporating real-time image and video analysis to enhance
accuracy and efficiency. Compared with the sole use of finer
models or simpler boundary features, using MobileNet along
with LSTM for feature extraction and sequential pattern
recognition increases the deepfake identification accuracy by
capturing spatial and temporal inconsistencies considerably.
The model makes a key innovation in being real time, which
makes it ideal for application in digital forensics and cyber
security. Nevertheless, it is limited by the computational
constraints of resource-constrained devices, the performance
degradation when dealing with very sophisticated deepfake
algorithms, and the need for enormous labeled datasets to
ensure robustness concerning different deepfake variations.
Continuously updating the model, however, the study
emphasizes that it is key to keep up to date with evolving
deepfake techniques in an effective way.


Bansal et al. [11] introduce a real-time advanced
computational intelligence framework for deepfake video
detection, leveraging machine learning and deep learning
techniques to enhance detection accuracy and efficiency. One
of the main contributions to the study is the combining of realtime processing facilities with attractive feature extraction
methods to make the identification of the deepfake
manipulations in dynamic video content more reliable. The
research further investigates the adoption of the adaptation
learning mechanism to improve the deepfake detection
models against changing deepfake attacks. The reported
limitations include high computational effort in real-time
processing, the possibility to drop in detection accuracy on a
variety of datasets (including natural ones and simulations),
and the inability of proficient deepfake evasion strategies
aimed to bypass detection. The study claims that deepfake
detection systems require continuous updates and powerful
training schemes to make them sound.


Heidari et al. [12] provide a systematic and comprehensive
review of deepfake detection techniques, focusing on deep
learning-based approaches and their effectiveness in
identifying manipulated media. The key innovation of the
study is its structured analysis of some deep learning
architectures, such as convolutional neural networks (CNN),
recurrent neural networks (RNN), and transformer-based
models, and their strengths and weaknesses in cases such as



deepfake detection. There is also a review of hybrid
methodologies and ensemble learning techniques that enhance
the detection robustness. Nevertheless, there are limitations of
generalizability to deepfake datasets different from the one
used in testing and the computational expense of training deep
learning models, as well as the development of the
sophisticated adversarial deepfake generation methods, which
could be uncaptured by current detection mechanisms. A
continuous cycle of model refinement and expansion of
dataset judiciously motivates the study’s finding, especially
the potential use in real-world indicators.


III. PROPOSED METHODOLOGY


Fig 1. System Architecture


_3.1_ _CNN-Based Deepfake Detection Framework_

A Convolutional Neural Network (CNN) is built for
spatial feature extraction of images and videos for deepfake
detection. The CNN model has multiple layers of
convolutions with ReLUs and max pooling layers that help to
preserve the spatial information and reduce the dimensionality.
Fully connected layers are applied to extracted features before
the final sigmoid activation function determines the
probability of an input being a deepfake [11]. To improve
detection robustness over subtle inconsistencies in
manipulated images, skip connections are applied to the
architecture. In comparison to the traditional CNN, our work
decreases overfitting using batch normalization and dropout
layers and is more easily applied to different deepfake
generation methods.


_3.2_ _Feature Extraction Process_

Deepfake detection is a critical challenge because
meaningful features that can discriminate real content from
manipulations by AI are extracted. To detect such fine-grained
inconsistencies as texture distortions, unnatural facial
expressions, and lighting irregularities, we employ a multiscale feature extraction. Deepfake detection is desired to have
spatial dependencies captured by CNNs, while Fourier and
wavelet transforms are utilized for frequency-based analysis
and discrepancies of deepfake videos. Finally, the hybrid



979-8-3315-0724-4/25/$31.00 ©2025 IEEE 135
Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:40:28 UTC from IEEE Xplore. Restrictions apply.


Proceedings of the Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS-2025)
IEEE Xplore Part Number: CFP25CB2-ART; ISBN: 979-8-3315-0724-4



feature fusion combines the CNN-derived features with the
transformer attention mechanisms to represent local as well as
global clues. These combined results yield great
improvements in model robustness concerning adversarial
attacks and compression artifacts that usually degrade
detection accuracy.


_3.3_ _Optimization and Loss Reduction Strategies_

To enhance model efficiency and accuracy, several
optimization strategies are implemented. The performance of
the model is trained by the Adam optimizer, which
dynamically changes learning rates to accelerate the
convergence speed. To deal with the class imbalance and
increase the classification confidence, the hybrid loss function
that combines the binary cross entropy loss with the focal loss
is introduced. Additionally, dropout regularization and weight
decay aid in avoiding overfitting thanks to stable
generalization when the deepfake dataset changes. Dynamic
training parameter (adaptive learning rate) scheduling adjusts
the training parameters to utilize the updates from the gradient
in an optimal learning efficient manner. Taken together, these
strategies make precision high, eliminate false positives, and
can maintain high detection accuracy when facing unseen
deepfakes.


_3.4_ _Integration with Transformers and Recurrent Networks_

Deepfake videos are temporal in the sense that they are
highly inconsistent in time. We take the approach of
integrating transformers and recurrent networks that help to
enhance sequential analysis. The vision transformers (ViTs)
are used to achieve significant long-range dependencies in
images while improving the feature representation networks,
and the long short-term memory (LSTM) network is used to
deal with the motion inconsistencies among frames [12]. By
combining the strengths of both approaches in a hybrid
fashion, we can detect such subtle deepfake artifacts, like
overly unnatural blinking patterns as the eyes did not blink
with a similar pattern as they do when alive, or when there are
obvious discrepancies in frames. The model uses CNNs
combined with transformers and LSTMs to obtain higher
robustness and adaptability compared to the more traditional
deepfake detection methods.


_3.5_ _Real-Time Processing Considerations_

For real-time applications, the balance between detection
accuracy and computational efficiency is important. To
minimize complexity with reasonable inference speed, model
pruning, quantization, and TensorRT acceleration are
incorporated into the framework. Taking advantage of GPU
acceleration and parallel processing, running even at high
resolution in real time, it also reduces latency, giving real-time
deepfake detection. Resource usage is a trade-off against the
models' complexity to give the best detections and models,
respectively. On-device inference techniques are also
explored for integrating the detection framework to mobile
and edge computing environments such that the system is
scalable for deployment under security, digital forensics, and
social media settings.


_3.6_ _Robustness Against Adversarial Attacks_

Adversarial attacks on deepfake detection models can
manipulate pixels to evade detection. They overcome this by
applying adversarial training techniques, which are used to
expose the model to perturbed deepfake examples for
improving robustness. Two of the approaches that are used are
defensive distillation and gradient masking to prevent



attackers from leveraging the vulnerabilities of models. Also,
contrastive learning techniques are used to separate real
features from adversarially changed ones to make deepfake
techniques more resilient to evolving. This ensures that the
deepfake framing is still able to handle novel, sophisticated
deepfake generation models.


_Algorithm: CNN-Based Deepfake Detection_

Step:1. Input a video frame or image for classification.


Step:2. Resize the image to 224×224 pixels and normalize

pixel values.


Step:3. Apply data augmentation techniques such as rotation,

flipping, and contrast adjustments.


Step:4. Extract features using multiple convolutional layers

with ReLU activation.


Step:5. Perform max-pooling for dimensionality reduction

and apply batch normalization.


Step:6. Flatten the extracted features and pass them through

fully connected layers.


Step:7. Use a sigmoid activation function for binary

classification.


Step:8. Train the model using the Adam optimizer with a

hybrid focal and binary cross-entropy loss.


Step:9. Update model weights using backpropagation for

improved accuracy.


Step:10. During inference, classify as deepfake if the output

probability > 0.5; otherwise, classify as real.


Step:11. Apply threshold tuning to refine predictions and

display confidence scores for interpretability.


IV. RESULTS AND PERFORMANCE EVALUATION


_4.1_ _Accuracy and Loss Analysis_

Accuracy and loss trends give us an idea about what the
model is learning. Validation accuracy stabilizes, and the
training accuracy is always increasing with epochs, which
means that we are effectively generalizing. Good things are
confirmed by a progressive decrease of the loss function, i.e.,
low loss/damage by minimal overfitting. The proposed CNNbased method is then compared with the existing models,
where it has been seen to outperform the traditional deepfake
detection frameworks. As illustrated in Fig. 2, both training
and validation accuracies show consistent improvement
across multiple training series, indicating stable model
learning.












|TABLE|1. TRAINING|AND VALIDATION A EPOCHS|ACCURACY &|& LOSS OVER 10|
|---|---|---|---|---|
|**Epoch**|<br>**Training**<br>**Accuracy (%)**|**Validation**<br>**Accuracy (%)**|**Training**<br>**Loss**|**Validation**<br>**Loss**|
|1|78.5|74.2|0.52|0.61|
|2|82.3|78.9|0.44|0.55|
|3|85.7|81.4|0.39|0.49|
|4|88.1|83.7|0.34|0.42|
|5|90.3|86.2|0.28|0.37|
|6|92.1|87.9|0.23|0.31|
|7|93.4|89.3|0.19|0.27|
|8|94.2|90.1|0.16|0.24|



979-8-3315-0724-4/25/$31.00 ©2025 IEEE 136
Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:40:28 UTC from IEEE Xplore. Restrictions apply.


Proceedings of the Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS-2025)
IEEE Xplore Part Number: CFP25CB2-ART; ISBN: 979-8-3315-0724-4











works better than these existing solutions, having accuracy,
precision, recall, and F1-score.


TABLE 3. PERFORMANCE COMPARISON WITH EXISTING MODELS




|Epoch|Training<br>Accuracy (%)|Validation<br>Accuracy (%)|Training<br>Loss|Validation<br>Loss|
|---|---|---|---|---|
|9|95.1|91.5|0.13|0.21|
|10|96.0|92.8|0.11|0.18|














|Model|Accuracy<br>(%)|Precision<br>(%)|Recall<br>(%)|F1-Score<br>(%)|
|---|---|---|---|---|
|EfficientNet|91.2|90.1|92.3|91.2|
|Xception|93.5|92.7|94.0|93.3|
|ResNet50|89.8|89.2|90.7|89.9|
|**Proposed CNN**<br>**Model**|**92.8**|**92.9**|**92.9**|**92.9**|







Fig 2. Graphical Representation of Training and Validation Accuracy


_4.2_ _Precision, Recall, and F1-Score Evaluation_

The model is evaluated on precision, recall, and F1 score
to strike the right balance between the correct detection of
deepfake images and the minimization of false positives. High
precision decreases the false positives, and high recall ensures
the detection of the deepfake images correctly. Compared with
other models, the proposed method offers a more balanced and
better performance in all three metrics.


TABLE 2. PRECISION, RECALL, AND F1-SCORE FOR DEEPFAKE

DETECTION



Fig 4. Graphical Representation of Performance Comparison with Existing
Models


As shown in Table 3 and Fig. 4. For real-time deployment feasibility, the

suggested model exhibits the lowest training time and inference
latency, as summarized in Table 4 and visualized in Fig. 5. Both
training and validation accuracies consistently improve across multiple
training series, indicating stable model learning.


_4.4_ _Computational Efficiency and Inference Time_

Its computational efficiency is very important as in realtime applications. The proposed CNN model is faster during
inference and has lower training time per epoch and, thus, is
suitable for real-time deepfake detection.


TABLE 4. TRAINING AND INFERENCE TIME ANALYSIS






|Class|Precision (%)|Recall (%)|F1-Score (%)|
|---|---|---|---|
|Real Image|94.3|91.8|93.0|
|Deepfake|91.5|94.1|92.8|











Fig 3. Graphical Representation of Precision, Recall, and F1-Score for
Deepfake Detection


_4.3_ _Comparative Analysis with Existing Models_

In order to demonstrate the effectiveness of the proposed
CNN-based approach, its performance has been compared
with those of popular deepfake detection models like
EfficientNet, Xception, and ResNet50. The proposed model




|Model<br>EfficientNet<br>Xception<br>ResNet50<br>Proposed CNN<br>Model|Training Time (per<br>epoch)|Inference Time (per<br>image)<br>15 ms<br>18 ms<br>20 ms<br>14 ms|
|---|---|---|
|**Model**<br>EfficientNet<br>Xception<br>ResNet50<br>**Proposed CNN**<br>**Model**|3.5 min|3.5 min|
|**Model**<br>EfficientNet<br>Xception<br>ResNet50<br>**Proposed CNN**<br>**Model**|4.2 min|4.2 min|
|**Model**<br>EfficientNet<br>Xception<br>ResNet50<br>**Proposed CNN**<br>**Model**|3.8 min|3.8 min|
|**Model**<br>EfficientNet<br>Xception<br>ResNet50<br>**Proposed CNN**<br>**Model**|**3.1 min**|**3.1 min**|



979-8-3315-0724-4/25/$31.00 ©2025 IEEE 137
Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:40:28 UTC from IEEE Xplore. Restrictions apply.


Proceedings of the Third International Conference on Augmented Intelligence and Sustainable Systems (ICAISS-2025)
IEEE Xplore Part Number: CFP25CB2-ART; ISBN: 979-8-3315-0724-4


[9] Lokhande, M., Raut, P., Gawali, K., Ahirrao, M., & Bhande, A. (2024,







Fig 5. Graphical Representation of Training and Inference Time Analysis


V. CONCLUSION

This research presents a CNN-based deepfake detection
framework that effectively identifies manipulated images with
high accuracy and efficiency. The proposed model adopts
deep convolutional layers to extract features, which brings
superior performance than the existing detection methods like
EfficientNet, Xception, and ResNet50. The optimization
strategies are also integrated with Adam optimizer and binary
cross entropy loss function to make learning efficient with
minimum false positives and false negatives. The model is
extensively experimented on and shows robustness in
detecting high-resolution deepfakes with high precision and
recall rates for a wide range of compression levels. The
proposed framework, in addition, has a high computational
efficiency that enables real-time detection applications.
Further improvement of detection accuracy can be achieved in
the future with a hybrid architecture using transformers or
recurrent networks. By providing a reliable solution to tackle
the rising risk of such deepfake technology, this study makes
an impact on the field of digital forensics and cybersecurity.


REFERENCES

[1] Nannaware, S. C., Pillai, R., & Kate, N. (2025). Deepfakes in Action:

Exploring Use Cases Across Industries. In _Deepfakes and Their Impact_
_on Business_ (pp. 71-98). IGI Global Scientific Publishing.

[2] Babaei, R., Cheng, S., Duan, R., & Zhao, S. (2025). Generative

Artificial Intelligence and the Evolving Challenge of Deepfake
Detection: A Systematic Analysis. _Journal of Sensor and Actuator_
_Networks_, _14_ (1), 17.

[3] Langa, J. (2021). Deepfakes, real consequences: Crafting legislation to

combat threats posed by deepfakes. _BUL Rev._, _101_, 761.

[4] Hasan, M., Athrey, K. S., Khalid, A., Xie, D., Younessian, E., &

Braskich, T. (2024). Applications of computer vision in the
entertainment and media industry. In _Computer Vision_ (pp. 205-238).
Chapman and Hall/CRC.


[5] Satone, K., & Amdani, S. Y. (2024, December). Preserving video

authenticity in the age of synthetic media using blockchain. In AIP
Conference Proceedings (Vol. 3188, No. 1). AIP Publishing.

[6] Sharma, S. K., AlEnizi, A., Kumar, M., Alfarraj, O., & Alowaidi, M.

(2024). Detection of real-time deep fakes and face forgery in video
conferencing employing generative adversarial networks. Heliyon,
10(17).

[7] Alrawahneh, A. A. M., Abdullah, S. N. A. S., Abdullah, S. N. H. S.,

Kamarudin, N. H., & Taylor, S. K. (2025). Video authentication
detection using deep learning: a systematic literature review. Applied
Intelligence, 55(3), 239.

[8] Sundaram, V., Senthil, B., & Vekkot, S. (2024, June). Enhancing

Deepfake Detection: Leveraging Deep Models for Video
Authentication. In _2024 15th International Conference on Computing_
_Communication and Networking Technologies (ICCCNT)_ (pp. 1-7).
IEEE.



August). Artificial Intelligence for Detecting Cyber Attacks in
Deepfake & Identity Theft. In _2024 8th International Conference on_
_Computing,_ _Communication,_ _Control_ _and_ _Automation_
_(ICCUBEA)_ (pp. 1-6). IEEE.

[10] Anandhasivam, V. S., Anusri, A. K., Logeshwar, M., & Gopinath, R.

(2024, December). Enhancing Deepfake Detection Through Hybrid
MobileNet-LSTM Model with Real-Time Image and Video Analysis.
In _2024 4th International Conference on Ubiquitous Computing and_
_Intelligent Information Systems (ICUIS)_ (pp. 1989-1995). IEEE.

[11] Bansal, N., Aljrees, T., Yadav, D. P., Singh, K. U., Kumar, A., Verma,

G. K., & Singh, T. (2023). Real-time advanced computational
intelligence for deepfake video detection. _Applied Sciences_, _13_ (5),
3095.

[12] Heidari, A., Jafari Navimipour, N., Dag, H., & Unal, M. (2024).

Deepfake detection using deep learning methods: A systematic and
comprehensive review. _Wiley Interdisciplinary Reviews: Data Mining_
_and Knowledge Discovery_, _14_ (2), e1520.

[13] Abbasi, M., Váz, P., Silva, J., & Martins, P. (2025). Comprehensive

Evaluation of Deepfake Detection Models: Accuracy, Generalization,
and Resilience to Adversarial Attacks. _Applied Sciences (2076-_
_3417)_, _15_ (3).



979-8-3315-0724-4/25/$31.00 ©2025 IEEE 138
Authorized licensed use limited to: Acharya Institute of Technology. Downloaded on April 13,2026 at 05:40:28 UTC from IEEE Xplore. Restrictions apply.


