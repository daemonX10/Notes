# Segmentation & Vision Transformers - Interview Questions

## Semantic Segmentation

### Question 1
**Explain U-Net's encoder-decoder architecture with skip connections. Why are skip connections crucial for segmentation?**

**Answer:** _To be filled_

---

### Question 2
**What are the key innovations in DeepLabv3+ (atrous convolutions, ASPP, encoder-decoder) for boundary delineation?**

**Answer:** _To be filled_

---

### Question 3
**Compare Dice loss vs. Cross-entropy for segmentation. When would you use focal loss or Lovász loss?**

**Answer:** _To be filled_

---

### Question 4
**How do atrous/dilated convolutions capture multi-scale context without losing resolution?**

**Answer:** _To be filled_

---

### Question 5
**Explain U-Net variants: U-Net++, Attention U-Net, and TransUNet. What problems does each solve?**

**Answer:** _To be filled_

---

### Question 6
**How do you handle class imbalance in segmentation when some classes occupy very few pixels?**

**Answer:** _To be filled_

---

### Question 7
**What techniques improve segmentation performance on small or thin objects (boundary loss, deep supervision)?**

**Answer:** _To be filled_

---

### Question 8
**Explain the overlap-tile strategy for inference on large images that don't fit in memory.**

**Answer:** _To be filled_

---

### Question 9
**How do you implement data augmentation that preserves spatial relationships for segmentation?**

**Answer:** _To be filled_

---

### Question 10
**What approaches work for real-time semantic segmentation (BiSeNet, Fast-SCNN, EfficientPS)?**

**Answer:** _To be filled_

---

### Question 11
**How do you handle segmentation of objects with fuzzy or ambiguous boundaries?**

**Answer:** _To be filled_

---

### Question 12
**Explain weakly supervised segmentation using image-level labels or bounding boxes instead of pixel masks.**

**Answer:** _To be filled_

---

## Instance Segmentation

### Question 13
**How does Mask R-CNN's architecture balance object detection and pixel-level segmentation?**

**Answer:** _To be filled_

---

### Question 14
**Explain the difference between ROIPool and ROIAlign. Why is ROIAlign crucial for mask quality?**

**Answer:** _To be filled_

---

### Question 15
**What are the trade-offs between two-stage (Mask R-CNN) and single-stage (YOLACT, SOLOv2) instance segmentation?**

**Answer:** _To be filled_

---

### Question 16
**How do you handle overlapping instances in dense object arrangements?**

**Answer:** _To be filled_

---

### Question 17
**Explain panoptic segmentation and how it combines semantic and instance segmentation.**

**Answer:** _To be filled_

---

### Question 18
**What techniques help with segmenting objects with complex or irregular shapes?**

**Answer:** _To be filled_

---

### Question 19
**How do you optimize Mask R-CNN for real-time applications without significant accuracy loss?**

**Answer:** _To be filled_

---

## Vision Transformers (ViT)

### Question 20
**Explain the core innovation of Vision Transformers compared to CNNs. What inductive biases does ViT lack?**

**Answer:** _To be filled_

---

### Question 21
**How are images converted into patch embeddings in ViT? Explain the linear projection layer.**

**Answer:** _To be filled_

---

### Question 22
**What is the role of the [CLS] token and positional encodings in Vision Transformers?**

**Answer:** _To be filled_

---

### Question 23
**Explain the computational complexity of ViT (O(n²) attention) and how it limits resolution scalability.**

**Answer:** _To be filled_

---

### Question 24
**What are the data requirements for training ViT from scratch vs. using pre-trained models?**

**Answer:** _To be filled_

---

### Question 25
**Explain DeiT (Data-efficient Image Transformers) and how knowledge distillation improves ViT training on smaller datasets.**

**Answer:** _To be filled_

---

### Question 26
**Describe Masked Autoencoder (MAE) pre-training for Vision Transformers. How does masking 75% of patches work?**

**Answer:** _To be filled_

---

### Question 27
**How do hybrid architectures combine CNN feature extraction with transformer attention?**

**Answer:** _To be filled_

---

### Question 28
**Explain attention visualization in ViT. How do you interpret attention patterns across layers?**

**Answer:** _To be filled_

---

### Question 29
**What are the architectural variants of ViT (ViT-B, ViT-L, ViT-H) and their trade-offs?**

**Answer:** _To be filled_

---

### Question 30
**How does ViT handle different input image resolutions during fine-tuning vs. pre-training?**

**Answer:** _To be filled_

---

## Swin Transformer

### Question 31
**Explain shifted window partitioning in Swin Transformer. How does it enable cross-window connections?**

**Answer:** _To be filled_

---

### Question 32
**How does Swin Transformer's hierarchical representation differ from ViT's flat structure?**

**Answer:** _To be filled_

---

### Question 33
**Explain the linear complexity (O(n)) of Swin vs. quadratic complexity of ViT. How is this achieved?**

**Answer:** _To be filled_

---

### Question 34
**Describe patch merging layers and how they create multi-scale feature maps.**

**Answer:** _To be filled_

---

### Question 35
**Explain relative positional bias in Swin vs. absolute positional encoding in ViT.**

**Answer:** _To be filled_

---

### Question 36
**How is Swin Transformer used as a backbone for object detection (Swin + FPN) and segmentation (Swin + UPerNet)?**

**Answer:** _To be filled_

---

### Question 37
**Compare Swin-T, Swin-S, Swin-B, and Swin-L configurations. How do you choose for your task?**

**Answer:** _To be filled_

---

### Question 38
**Explain Swin-V2 improvements: log-scaled continuous position bias, residual post-norm, and scaled cosine attention.**

**Answer:** _To be filled_

---

### Question 39
**How does Video Swin Transformer extend the architecture for temporal modeling?**

**Answer:** _To be filled_

---

### Question 40
**Compare Swin Transformer to ConvNeXt. What design principles from Swin were adopted back into CNNs?**

**Answer:** _To be filled_

---

## Segmentation Transformers

### Question 41
**Explain SETR (Segmentation Transformer) and how pure transformers handle dense prediction.**

**Answer:** _To be filled_

---

### Question 42
**Describe SegFormer architecture and its efficient self-attention mechanism for segmentation.**

**Answer:** _To be filled_

---

### Question 43
**How does Mask2Former unify semantic, instance, and panoptic segmentation with a single architecture?**

**Answer:** _To be filled_

---

### Question 44
**Explain SAM (Segment Anything Model) and its promptable segmentation capabilities.**

**Answer:** _To be filled_

---

## Practical Considerations

### Question 45
**How do you handle temporal consistency in video semantic/instance segmentation?**

**Answer:** _To be filled_

---

### Question 46
**What approaches work for domain adaptation in segmentation across different imaging modalities?**

**Answer:** _To be filled_

---

### Question 47
**Explain active learning strategies for efficient mask annotation in segmentation tasks.**

**Answer:** _To be filled_

---

### Question 48
**How do you implement uncertainty quantification in segmentation predictions?**

**Answer:** _To be filled_

---

### Question 49
**What techniques help segment objects in adverse weather or lighting conditions?**

**Answer:** _To be filled_

---

### Question 50
**How do you handle segmentation with limited computational resources or memory on edge devices?**

**Answer:** _To be filled_

---

## Evaluation & Medical Imaging

### Question 51
**Explain IoU, Dice coefficient, and boundary F1-score for segmentation evaluation. When to use each?**

**Answer:** _To be filled_

---

### Question 52
**How do you optimize U-Net architectures for medical image segmentation (3D U-Net, nnU-Net)?**

**Answer:** _To be filled_

---

### Question 53
**What are the challenges of segmentation in specialized domains like satellite imagery or microscopy?**

**Answer:** _To be filled_

---

### Question 54
**Explain federated learning for medical image segmentation across hospitals with privacy constraints.**

**Answer:** _To be filled_

---

### Question 55
**How do you handle noisy or inconsistent annotations in segmentation ground truth?**

**Answer:** _To be filled_

---

## Advanced Topics

### Question 56
**Explain few-shot segmentation in novel semantic categories without retraining.**

**Answer:** _To be filled_

---

### Question 57
**How do you implement knowledge distillation for compressing large segmentation models?**

**Answer:** _To be filled_

---

### Question 58
**What techniques help with explaining segmentation decisions for model interpretability?**

**Answer:** _To be filled_

---

### Question 59
**How do you integrate conditional random fields (CRF) as post-processing for segmentation refinement?**

**Answer:** _To be filled_

---

### Question 60
**Explain multi-task learning that combines segmentation with depth estimation or other vision tasks.**

**Answer:** _To be filled_

---
