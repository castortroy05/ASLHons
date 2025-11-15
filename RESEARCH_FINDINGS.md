# ASL Recognition & Translation - Research Findings 2024-2025

## Executive Summary
This document summarizes state-of-the-art approaches for ASL recognition and text-to-sign translation based on current research and available tools.

---

## 1. Pre-trained Sign Language Recognition Models

### MediaPipe Hand Landmark Detection
- **Developer**: Google
- **Capability**: Real-time hand tracking with 21 keypoints per hand
- **Performance**: Industry standard for hand pose estimation
- **Use Case**: Feature extraction for downstream ASL recognition
- **Integration**: Python SDK available, works with TensorFlow/PyTorch

### Recent SOTA Models (2024-2025)

#### YOLOv11 + MediaPipe (March 2025)
- **Performance**: 98.2% mAP@0.5 for ASL alphabet
- **Approach**: YOLOv11 for gesture detection + MediaPipe for hand tracking
- **Advantage**: Real-time performance with high accuracy

#### KD-MSLRT (January 2025)
- **Innovation**: Knowledge distillation for lightweight deployment
- **Target**: Resource-constrained environments (mobile/edge)
- **Features**: 3D to 1D knowledge distillation based on MediaPipe

#### SLRNet (June 2025)
- **Architecture**: MediaPipe Holistic + LSTM
- **Performance**: 86.7% validation accuracy for dynamic gestures
- **Features**: Extracts 543 landmarks per frame for comprehensive tracking

#### Stanford CS231n 2024
- **Approach**: MediaPipe landmarks + CNN
- **Performance**: 99.12% accuracy on ASL datasets
- **Method**: Pre-trained hand landmark model + custom CNN classifier

---

## 2. Text-to-Sign Language Avatar Systems

### GenASL (AWS - 2024)
- **Repository**: https://github.com/aws-samples/genai-asl-avatar-generator
- **Technology Stack**:
  - Amazon Transcribe (speech-to-text)
  - Amazon SageMaker (model training)
  - Amazon Bedrock (generative AI)
  - Foundation Models for animation
- **Input**: Audio, video, or text
- **Output**: ASL avatar video with expressive animations
- **Status**: Open-source, production-ready
- **Licensing**: AWS open-source license

### SignAvatar (2024)
- **Website**: https://www.signavatar.org/
- **Languages**: 30+ including ASL, International Sign Language
- **Integration**: Works as software layer on existing systems
- **Features**: Real-time translation with avatar visualization

### SiMAX
- **Innovation**: First 3D animated avatar system for sign language
- **Capability**: Translates text into multiple sign languages
- **Target**: Digital accessibility for deaf community

### SignAvatars (ECCV 2024)
- **Type**: Research dataset and benchmark
- **Repository**: https://github.com/ZhengdiYu/SignAvatars
- **Contribution**: Large-scale 3D sign language holistic motion dataset
- **Use Case**: Training avatar generation models

---

## 3. Best ASL Fingerspelling Datasets (2024)

### FSboard (2024) - LARGEST
- **Size**: 3+ million characters, 250+ hours
- **Scale**: 10x larger than previous largest dataset
- **Signers**: 147 paid and consenting Deaf signers
- **Collection**: Pixel 4A selfie cameras, various environments
- **Advantage**: Massive scale for modern deep learning
- **Diversity**: Real-world environmental variations

### ChicagoFSWild+ (Large-scale)
- **Sequences**: 55,232 fingerspelling sequences
- **Signers**: 260 different signers
- **Source**: YouTube and Deaf social media ("in the wild")
- **Annotation**: Carefully annotated by ASL students
- **Advantage**: Natural, realistic fingerspelling
- **Baseline**: ChicagoFSWild has 7,304 sequences from 160 signers

### ASL Fingerspelling Dataset (Pugeault & Bowden, 2011)
- **Size**: 64,000+ hand-cropped images
- **Modality**: RGB + Depth (Microsoft Kinect)
- **Letters**: 24 static ASL letters
- **Advantage**: Depth information for 3D understanding
- **Status**: Historical baseline, still used for comparisons

### IEEE DataPort ASL Dataset (May 2024)
- **DOI**: 10.21227/cbg0-7552
- **Format**: Multiple image formats
- **Access**: IEEE DataPort (may require membership)
- **Recency**: Very recent (May 2024)

### MiCT-RANet-ASL-FingerSpelling
- **Repository**: https://github.com/fmahoudeau/MiCT-RANet-ASL-FingerSpelling
- **Performance**: 74.4% letter accuracy on ChicagoFSWild+
- **Features**: Real-time fingerspelling video recognition
- **Innovation**: Multi-scale temporal attention

---

## 4. Recommended Architecture for This Project

### Hybrid Approach: Sign-to-Text + Text-to-Sign

#### Sign-to-Text Recognition Pipeline
1. **Hand Detection**: MediaPipe Hand Landmark Detection
   - Extract 21 keypoints per hand
   - Real-time performance on CPU/GPU

2. **Feature Extraction**:
   - Option A: EfficientNetV2 (lightweight, high accuracy)
   - Option B: MobileNetV3 (for edge deployment)
   - Option C: Custom CNN on MediaPipe landmarks only

3. **Classification**:
   - For static letters: Simple MLP or CNN
   - For dynamic signs: LSTM/GRU/Transformer on temporal sequences

4. **Training Strategy**:
   - Transfer learning with frozen base
   - Fine-tune only top layers
   - Person-based train/test split
   - Heavy data augmentation

#### Text-to-Sign Translation Pipeline
1. **Text Processing**:
   - Tokenization and normalization
   - Convert to ASL gloss (ASL grammar differs from English)

2. **Sign Sequence Generation**:
   - Look up static fingerspelling images
   - Retrieve sign videos from database

3. **Avatar Animation**:
   - Option A: Use GenASL (AWS solution) if cloud-based
   - Option B: Create simple stick-figure avatar from MediaPipe landmarks
   - Option C: Use pre-recorded video sequences

4. **Rendering**:
   - OpenCV for video composition
   - Matplotlib for static frames
   - Unity/Blender for 3D avatars (advanced)

---

## 5. Implementation Priority

### Phase 1: Fix Existing Model (Week 1-2)
- [ ] Fix train/val/test split methodology
- [ ] Add person-based splitting if metadata available
- [ ] Implement data augmentation
- [ ] Use proper input size (224x224) or switch to EfficientNet
- [ ] Freeze base layers appropriately
- [ ] Add comprehensive metrics and visualization

### Phase 2: MediaPipe Integration (Week 2-3)
- [ ] Install MediaPipe Python SDK
- [ ] Create landmark extraction pipeline
- [ ] Train model on landmarks only (baseline)
- [ ] Train hybrid model (landmarks + images)
- [ ] Compare performance

### Phase 3: Text-to-Sign Basic (Week 3-4)
- [ ] Create fingerspelling database (letter → image mapping)
- [ ] Implement text → letter sequence conversion
- [ ] Create video/GIF generation from letter sequence
- [ ] Build simple visualization (matplotlib/opencv)

### Phase 4: Avatar System (Week 4-6)
- [ ] Research GenASL integration OR
- [ ] Build custom stick-figure avatar from landmarks
- [ ] Create smooth transitions between signs
- [ ] Add facial expressions (if using 3D avatar)

### Phase 5: Production System (Week 6-8)
- [ ] Create unified Streamlit/Gradio web app
- [ ] Integrate all components (sign→text, text→sign)
- [ ] Deploy to cloud (AWS/Hugging Face/local)
- [ ] Create documentation and demo video

---

## 6. Key Technical Improvements Over Original

| Aspect | Original | Improved |
|--------|----------|----------|
| **Input Size** | 64x64 (too small) | 224x224 or landmarks |
| **Data Split** | Random, wrong validation | Stratified, person-based |
| **Augmentation** | None | Rotation, zoom, brightness |
| **Base Model** | VGG19 (all trainable) | EfficientNet/MobileNet (frozen) |
| **Feature Extraction** | Raw pixels only | MediaPipe landmarks + pixels |
| **Validation** | From test set! | From training set |
| **Metrics** | Accuracy only | Precision, recall, F1, confusion matrix |
| **Functionality** | One-way (sign→text) | Bidirectional (sign↔text) |
| **Avatar** | None | 3D/2D avatar visualization |
| **Deployment** | Jupyter notebook only | Web app + API |

---

## 7. Useful Resources

### Code Repositories
- MediaPipe Python: `pip install mediapipe`
- GenASL AWS: https://github.com/aws-samples/genai-asl-avatar-generator
- MiCT-RANet: https://github.com/fmahoudeau/MiCT-RANet-ASL-FingerSpelling
- SignAvatars: https://github.com/ZhengdiYu/SignAvatars

### Papers
- "MediaPipe Hands: On-device Real-time Hand Tracking" (Google, 2020)
- "Sign Language Recognition with Convolutional Neural Networks" (Stanford CS231n, 2024)
- "KD-MSLRT: Lightweight Sign Language Recognition" (arXiv, 2025)

### Datasets
- FSboard: Contact authors (latest, largest)
- ChicagoFSWild+: Publicly available
- Current dataset: ../ASLTransalation/fingerspelling/data/ (68K images)

---

## 8. Expected Outcomes

### Realistic Performance Targets
- **Static letter recognition**: 95-98% (with proper methodology)
- **Fingerspelling sequences**: 70-85% letter accuracy
- **Real-time inference**: 30+ FPS on GPU, 10+ FPS on CPU
- **Text-to-sign**: 100% accuracy (deterministic)

### Success Criteria
1. **Sign Recognition**: Generalizes to new users not in training set
2. **Text-to-Sign**: Smooth, understandable avatar animations
3. **System Integration**: End-to-end pipeline working in real-time
4. **Code Quality**: Modular, documented, reproducible
5. **Academic Rigor**: Proper experimental methodology

---

Last Updated: 2025-01-15
