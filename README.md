# Joint Holistic and Masked Face Recognition

![ov](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/overview.png)

The official implement of paper ''[Joint Holistic and Masked Face Recognition](https://ieeexplore.ieee.org/abstract/document/10138097)''.

## Abstract 

><div align="justify">With the widespread use of face masks due to the COVID-19 pandemic, accurate masked face recognition has become more crucial than ever. While several studies have investigated masked face recognition using convolutional neural networks (CNNs), there is a paucity of research exploring the use of plain Vision Transformers (ViTs) for this task. Unlike ViT models used in image classification, object detection, and semantic segmentation, the model trained by modern face recognition losses struggles to converge when trained from scratch. To this end, this paper initializes the model parameters via a proxy task of patch reconstruction and observes that the ViT backbone exhibits improved training stability with satisfactory performance for face recognition. Beyond the training stability, two strategies based on prompts are proposed to integrate holistic and masked face recognition in a single framework, namely FaceT. Along with popular holistic face recognition benchmarks, several open-sourced masked face recognition benchmarks are collected for evaluation. Our extensive experiments demonstrate that the proposed FaceT performs on par or better than state-of-the-art CNNs on both holistic and masked face recognition benchmarks.</div>

------

## Training
Please see this [README](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/code/README.md) file for details.

## Benchmarks & Results
All models are trained on WebFace42M, where 10% faces are randomly masked by a face masking tool, FMA-3D, from [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo).

### Holistic Benchmarks
Models are tested on popular holistic face recognition benchmarks, including LFW, AgeDB-30, CFP-FP, CALFW, CPLFW, RFW, IJB-B, and IJB-C for holistic face verification; and on MegaFace for both holistic face identification and verification.

![t1](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/t1.PNG)

In this table, $^{\pm}$ denotes the model trained on pure holistic data and the best available results are quoted from previous works. Results include verification accuracy on LFW, CFP-FP, AGEDB-30, CALFW, and CPLFW. RFW denotes the mean accuracy among Africans, Asians, Caucasians, and Indians. The column "Mean" refers to the mean accuracy among previous columns for comparison simplicity. When evaluating on MegaFace,  "Id"' refers to the rank-1 for identification rate with 1M distractors and "Ver" refers to the face verification TAR@FPR=1e-6. Finally, verification TAR@FPR=1e-4 is reported on IJB-B and IJB-C. 


### Masked Benchmarks
Masked LFW, Masked AGEDB-30, and Masked CFP-FP introduced by [Simulated Masked Face Recognition Datasets](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset). The masked face images are resized to $112 \times 112$ and the test protocol rigidly follows the original version of LFW, AGEDB-30, and CFP-FP. These benchmarks are synthesized for masked face verification.

[Real-world Masked Face Verification Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset). It contains 4015 face images of 426 people, which are organized into 7178 masked and non-masked sample pairs, including 3589 pairs of the same identity and 3589 pairs of different identities. The dataset contains real masked faces for verification.

[PKU-Masked-Face](https://pkuml.org/resources/pku-masked-face-dataset.html) contains 10,301 face images of 1,018 identities, each of which has masked and holistic face images with various orientations and mask types. Following their experiments, the frontal holistic faces are used as the gallery and frontal masked faces are used as the probe set. The identification performance is measured by the rank-1 accuracy of the masked faces.

[MegaFace-Mask](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/test_protocol) provides the pair list and tools for masking faces in the probe set (Facescrub) with 8 different types of masks. And the gallery set (MegaFace distractors) remains holistic. The benchmark is built for both masked face verification and identification.

![t3](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/t3.PNG)

The verification accuracy on Masked-LFW, Masked CFP-FP, and Masked AGEDB-30 are averaged in the Column "SMFRD". The verification accuracy evaluated on RMFVD and the rank-1 identification rate tested on PKU-Masked-Face are reported as follows. When evaluating on MegaFace-Mask,  "`Id" refers to the rank-1 for identification rate with 1M distractors and "Ver" refers to the face verification TAR@FPR=1e-6.
