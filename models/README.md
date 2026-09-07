# EAGLE-MARS

## Industrial Video-Based Fault Detection Using VideoMAE

EAGLE-MARS is a research project for automated fault detection in industrial machinery using short video sequences and transformer-based video classification.

The project investigates whether temporal visual information from machine-operation videos can be used to distinguish between:

- **NORMAL** machine operation
- **FAULT** conditions

The main deep-learning architecture used in the project is **VideoMAE**, with experiments focused on short temporal windows of video frames.

The project is being developed as part of a Master's thesis with a focus on industrial computer vision, video understanding, fault detection, and continual learning.

---

## 1. Project Objective

The objective of EAGLE-MARS is to develop a video-based machine fault detection pipeline capable of analyzing industrial machine operation and identifying abnormal behavior.

Instead of classifying individual images, the system processes a sequence of consecutive video frames so that temporal information about machine movement and operation can be considered.

The overall workflow is:

```text
Industrial Videos
       │
       ▼
CVAT Annotations
       │
       ▼
Annotation / Video Mapping
       │
       ▼
Temporal Fault Intervals
       │
       ▼
Sliding Video Windows
       │
       ▼
VideoMAE
       │
       ▼
NORMAL / FAULT
       │
       ▼
Evaluation / Visualization