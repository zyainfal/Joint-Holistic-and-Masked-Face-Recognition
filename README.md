# Joint-Holistic-and-Masked-Face-Recognition

The official implementation of the paper Joint Holistic and Masked Face Recognition.

The paper is currently under review and codes will be released upon acceptance.

## Abstract 

><div align="justify">With the widespread use of face masks due to the COVID-19 pandemic, accurate masked face recognition has become more crucial than ever. Yet some works have investigated masked face recognition by using convolutional neural networks (CNNs), these methods ineluctably trade-off between holistic and masked face recognition because of the feature fusion operations used in CNNs. On the contrary, plain Vision Transformers (ViTs) extract features without spatial down-sampling and have shown very competitive performance for a wide range of vision applications, such as image classification, object detection, and semantic segmentation but seldom in face recognition. Indeed, ViTs trained from scratch tend to diverge under the supervision of modern face recognition losses. To this end, this paper proposes a plain Transformer-based backbone for face recognition with enhanced training stability and better performance. Based on the improved backbone, two strategies are proposed to unify the holistic and masked face recognition as a single framework, namely FaceT. Along with the popular holistic face recognition benchmarks, several open-sourced masked face recognition benchmarks are collected for validation. Exhaustive experiments demonstrate that the proposed FaceT performs on par or better than state-of-the-art CNNs on both holistic and masked face recognition benchmarks.</div>

------

## Benchmarks & Results
### Holistic Data
Models are tested on popular holistic face recognition benchmarks, including LFW, AgeDB-30, CFP-FP, CALFW, CPLFW, RFW, IJB-B, and IJB-C for holistic face verification; and on MegaFace for both holistic face identification and verification.

![t1](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/t1.PNG)

In this table, $^{\pm}$ denotes the model trained on pure holistic data and the best available results are quoted from previous works. Results include verification accuracy on LFW, CFP-FP, AGEDB-30, CALFW, and CPLFW. RFW denotes the mean accuracy among Africans, Asians, Caucasians, and Indians. The column "Mean" refers to the mean accuracy among previous columns for comparison simplicity. When evaluating on MegaFace,  "Id"' refers to the rank-1 for identification rate with 1M distractors and "Ver" refers to the face verification TAR@FPR=1e-6. Finally, verification TAR@FPR=1e-4 is reported on IJB-B and IJB-C. Please note that previous models have not been tested on masked face recognition benchmarks.

### Masked Data
Masked LFW, Masked AGEDB-30, and Masked CFP-FP introduced by [Simulated Masked Face Recognition Datasets](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset). The masked face images are resized to $112 \times 112$ and the test protocol rigidly follows the original version of LFW, AGEDB-30, and CFP-FP. These benchmarks are synthesized for masked face verification.

[Real-world Masked Face Verification Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset). It contains 4015 face images of 426 people, which are organized into 7178 masked and non-masked sample pairs, including 3589 pairs of the same identity and 3589 pairs of different identities. The dataset contains real masked faces for verification.

[PKU-Masked-Face](https://pkuml.org/resources/pku-masked-face-dataset.html) contains 10,301 face images of 1,018 identities, each of which has masked and holistic face images with various orientations and mask types. Following their experiments, the frontal holistic faces are used as the gallery and frontal masked faces are used as the probe set. The identification performance is measured by the rank-1 accuracy of the masked faces.

[MegaFace-Mask](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/test_protocol), provides the pair list and tools for masking faces in the probe set, Facescrub with 8 different types of masks. And the gallery set, MegaFace distractors, remains holistic. The benchmark is built for both masked face verification and identification.

![t2](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/t2.PNG)

The verification accuracy on Masked-LFW, Masked CFP-FP, and Masked AGEDB-30 are averaged in the Column "SMFRD". The verification accuracy evaluated on RMFVD and the rank-1 identification rate tested on PKU-Masked-Face are reported as follows. When evaluating on MegaFace-Mask,  "`Id" refers to the rank-1 for identification rate with 1M distractors and "Ver" refers to the face verification TAR@FPR=1e-6.
