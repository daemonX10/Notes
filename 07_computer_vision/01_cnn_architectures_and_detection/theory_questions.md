# CNN Architectures, Classification & Object Detection - Interview Questions

## CNN Architecture Fundamentals

### Question 1
**Explain the vanishing gradient problem in deep CNNs and how ResNet skip connections solve it.**

**Answer:** _To be filled_

---

### Question 2
**Compare the architectural differences between ResNet-18, ResNet-50, and ResNet-101. When would you choose each?**

**Answer:** _To be filled_

---

### Question 3
**Explain bottleneck design in deeper ResNets and why it's more efficient than standard residual blocks.**

**Answer:** _To be filled_

---

### Question 4
**What is VGG's 3×3 convolution design philosophy and how does stacking small filters achieve larger receptive fields?**

**Answer:** _To be filled_

---

### Question 5
**Explain EfficientNet's compound scaling law (width, depth, resolution) and why balanced scaling outperforms single-dimension scaling.**

**Answer:** _To be filled_

---

### Question 6
**Describe Mobile Inverted Bottleneck Convolution (MBConv) and its role in efficient architectures.**

**Answer:** _To be filled_

---

### Question 7
**Explain depthwise separable convolutions in MobileNet and calculate the computational savings vs. standard convolutions.**

**Answer:** _To be filled_

---

### Question 8
**Compare MobileNet-v1, v2, and v3 architectural improvements. What are inverted residuals and linear bottlenecks?**

**Answer:** _To be filled_

---

### Question 9
**Explain Squeeze-and-Excitation (SE) blocks and how channel attention improves model performance.**

**Answer:** _To be filled_

---

### Question 10
**What is the dense connectivity pattern in DenseNet? Compare feature reuse in DenseNet vs. ResNet.**

**Answer:** _To be filled_

---

### Question 11
**Explain the Inception module with multiple kernel sizes and how 1×1 convolutions reduce dimensionality.**

**Answer:** _To be filled_

---

### Question 12
**Describe factorized convolutions (e.g., 7×7 into 1×7 and 7×1) and their computational benefits in InceptionNet.**

**Answer:** _To be filled_

---

### Question 13
**Explain ResNeXt's cardinality concept and how grouped convolutions improve accuracy over wider/deeper networks.**

**Answer:** _To be filled_

---

### Question 14
**Compare computational FLOPs, memory usage, and accuracy across ResNet, EfficientNet, and MobileNet for edge deployment.**

**Answer:** _To be filled_

---

### Question 15
**Explain knowledge distillation for compressing large CNN models to mobile-friendly versions.**

**Answer:** _To be filled_

---

## Image Classification

### Question 16
**How do you handle class imbalance in image classification beyond simple oversampling (focal loss, class weights, augmentation)?**

**Answer:** _To be filled_

---

### Question 17
**Explain fine-grained image classification techniques when inter-class differences are minimal.**

**Answer:** _To be filled_

---

### Question 18
**How do you implement cost-sensitive learning when misclassification costs vary across classes?**

**Answer:** _To be filled_

---

### Question 19
**What techniques improve model interpretability in medical image classification (Grad-CAM, attention, saliency maps)?**

**Answer:** _To be filled_

---

### Question 20
**Explain self-supervised pre-training strategies (contrastive learning, MAE) for image classification with limited labels.**

**Answer:** _To be filled_

---

### Question 21
**How do you design ensemble methods balancing accuracy and computational efficiency for classification?**

**Answer:** _To be filled_

---

### Question 22
**What are best practices for handling noisy labels in large-scale image classification datasets?**

**Answer:** _To be filled_

---

### Question 23
**Explain curriculum learning strategies for progressively training image classifiers on hard examples.**

**Answer:** _To be filled_

---

### Question 24
**How do you handle adversarial attacks in classification models (adversarial training, certified defenses)?**

**Answer:** _To be filled_

---

## Object Detection (YOLO, R-CNN Family)

### Question 25
**What are the trade-offs between single-stage (YOLO, SSD) and two-stage (Faster R-CNN) detectors?**

**Answer:** _To be filled_

---

### Question 26
**Explain the YOLO architecture evolution from v1 to v10. What are the key innovations in recent versions?**

**Answer:** _To be filled_

---

### Question 27
**How does anchor-free detection work in modern YOLO versions compared to anchor-based approaches?**

**Answer:** _To be filled_

---

### Question 28
**Discuss the R-CNN family evolution (R-CNN → Fast R-CNN → Faster R-CNN). What bottlenecks did each version solve?**

**Answer:** _To be filled_

---

### Question 29
**How do Feature Pyramid Networks (FPN) enable multi-scale object detection? Explain the top-down pathway.**

**Answer:** _To be filled_

---

### Question 30
**What techniques improve YOLO's performance on small object detection?**

**Answer:** _To be filled_

---

### Question 31
**Explain IoU, GIoU, DIoU, and CIoU losses. How do they improve bounding box regression?**

**Answer:** _To be filled_

---

### Question 32
**How do you handle class imbalance in object detection (focal loss, OHEM, class-balanced sampling)?**

**Answer:** _To be filled_

---

### Question 33
**Explain Non-Maximum Suppression (NMS) and its variants (Soft-NMS, DIoU-NMS). How does YOLOv10 eliminate NMS?**

**Answer:** _To be filled_

---

### Question 34
**What data augmentation strategies are specific to object detection (mosaic, mixup, copy-paste)?**

**Answer:** _To be filled_

---

### Question 35
**How do you detect objects with extreme aspect ratios or significant pose variations?**

**Answer:** _To be filled_

---

### Question 36
**Explain hard negative mining and its importance in training detection models.**

**Answer:** _To be filled_

---

### Question 37
**How do you handle detection of partially occluded or heavily crowded objects?**

**Answer:** _To be filled_

---

### Question 38
**What approaches work for few-shot object detection in novel categories?**

**Answer:** _To be filled_

---

### Question 39
**Explain domain adaptation techniques when deploying detection models to new environments.**

**Answer:** _To be filled_

---

### Question 40
**How do you optimize YOLO for real-time edge deployment (quantization, pruning, TensorRT)?**

**Answer:** _To be filled_

---

## Model Optimization & Deployment

### Question 41
**Compare quantization effects on different CNN architectures. Which are most quantization-friendly?**

**Answer:** _To be filled_

---

### Question 42
**Explain pruning strategies for CNNs (structured vs. unstructured, magnitude-based, lottery ticket hypothesis).**

**Answer:** _To be filled_

---

### Question 43
**What are the hardware acceleration considerations for GPU vs. TPU vs. NPU deployment?**

**Answer:** _To be filled_

---

### Question 44
**How do you implement batch normalization for inference vs. training? Explain folding BN into conv layers.**

**Answer:** _To be filled_

---

### Question 45
**Explain progressive resizing training strategy and its benefits for efficiency and accuracy.**

**Answer:** _To be filled_

---

## Evaluation & Metrics

### Question 46
**Explain mAP calculation for object detection. What's the difference between COCO mAP and Pascal VOC mAP?**

**Answer:** _To be filled_

---

### Question 47
**How do you handle detection evaluation when objects can have multiple valid annotations?**

**Answer:** _To be filled_

---

### Question 48
**What metrics beyond mAP are important for real-world detection systems (latency, false positive rate)?**

**Answer:** _To be filled_

---

## Advanced Topics

### Question 49
**Explain DETR (Detection Transformer) and how it differs from CNN-based detectors.**

**Answer:** _To be filled_

---

### Question 50
**How do you implement object tracking using detection-based approaches (DeepSORT, ByteTrack)?**

**Answer:** _To be filled_

---

### Question 51
**Explain attention mechanisms in object detection (CBAM, ECA, self-attention in detection heads).**

**Answer:** _To be filled_

---

### Question 52
**How do you design architectures that handle both common and rare object classes effectively?**

**Answer:** _To be filled_

---

### Question 53
**What techniques help with detecting objects in adverse weather conditions (fog, rain, low-light)?**

**Answer:** _To be filled_

---

### Question 54
**Explain uncertainty quantification in detection predictions and when it's important.**

**Answer:** _To be filled_

---

### Question 55
**How do you implement active learning for efficient annotation of detection datasets?**

**Answer:** _To be filled_

---
