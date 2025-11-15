# ASL Recognition & Translation System

A comprehensive American Sign Language (ASL) fingerspelling recognition and translation system with bidirectional capabilities.

## 🎯 Project Overview

This project provides:
- **Sign-to-Text**: Recognize ASL fingerspelling from images/webcam
- **Text-to-Sign**: Convert English text to ASL visualizations
- **Avatar Visualization**: Animated stick figure performing signs
- **MediaPipe Integration**: Advanced hand landmark detection
- **Modern Deep Learning**: EfficientNetV2 with proper transfer learning

## 🆕 Major Improvements Over Original

### Critical Fixes
1. ✅ **Proper Train/Val/Test Split** - Validation set now correctly created from training data, not test data
2. ✅ **Stratified Sampling** - Handles class imbalance correctly
3. ✅ **Correct Input Size** - 224×224 instead of 64×64 for transfer learning
4. ✅ **Frozen Base Layers** - Proper transfer learning instead of training all 20M+ parameters
5. ✅ **Data Augmentation** - Improves generalization (with ASL-appropriate transforms)
6. ✅ **Early Stopping** - Prevents overfitting
7. ✅ **Comprehensive Metrics** - Precision, recall, F1-score, confusion matrix

### New Features
- 🔥 **MediaPipe Hand Landmarks** - State-of-the-art hand tracking
- 🔥 **Text-to-Sign Translation** - Convert text to ASL
- 🔥 **Stick Figure Avatar** - Animated sign language visualization
- 🔥 **Interactive Web Demo** - Streamlit app for easy use
- 🔥 **Modular Architecture** - Clean, maintainable code structure

## 📁 Project Structure

```
ASLHons/
├── configs/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset_loader.py    # Proper data loading & splitting
│   │   └── augmentation.py      # Data augmentation
│   ├── models/
│   │   ├── model_builder.py     # Model architectures
│   │   └── mediapipe_extractor.py  # MediaPipe integration
│   ├── training/
│   │   └── evaluator.py         # Comprehensive evaluation
│   ├── text_to_sign/
│   │   └── translator.py        # Text-to-sign conversion
│   ├── avatar/
│   │   └── stick_figure_avatar.py  # Avatar visualization
│   └── utils/
│       ├── config_loader.py     # Config management
│       └── seed.py              # Reproducibility utilities
├── notebooks/                   # Jupyter notebooks
├── outputs/                     # Model outputs, metrics, videos
├── train_improved.py            # Training script
├── demo_app.py                  # Streamlit demo app
├── requirements.txt             # Dependencies
└── RESEARCH_FINDINGS.md         # Research documentation

```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ASLHons

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; import mediapipe as mp; print('✓ Setup complete')"
```

### Training the Model

```bash
# Train with default config
python train_improved.py

# Train with custom config
python train_improved.py --config configs/my_config.yaml

# Train with MediaPipe landmarks
python train_improved.py --use-mediapipe

# Preview data augmentation
python train_improved.py --preview-augmentation
```

### Running the Demo App

```bash
# Launch Streamlit app
streamlit run demo_app.py

# The app will open in your browser at http://localhost:8501
```

### Using Individual Components

#### Sign Recognition
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('outputs/checkpoints/best_model.h5')

# Load and preprocess image
img = cv2.imread('path/to/sign.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0

# Predict
prediction = model.predict(np.expand_dims(img, axis=0))
letter = chr(ord('a') + np.argmax(prediction))

print(f"Predicted letter: {letter}")
```

#### Text-to-Sign Translation
```python
from src.text_to_sign.translator import TextToSignTranslator

# Initialize translator
translator = TextToSignTranslator(
    letter_image_dir='data/processed/letter_templates'
)

# Create fingerspelling image
translator.create_fingerspelling_image(
    text="HELLO",
    output_path="outputs/hello.png"
)

# Create fingerspelling video
translator.create_fingerspelling_video(
    text="HELLO WORLD",
    output_path="outputs/hello_world.mp4"
)
```

#### Avatar Visualization
```python
from src.avatar.stick_figure_avatar import StickFigureAvatar
import glob

# Load letter images
letter_images = {}
for path in glob.glob('data/processed/letter_templates/*.png'):
    letter = Path(path).stem
    img = cv2.imread(path)
    letter_images[letter] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create avatar
with StickFigureAvatar(color_scheme='colorful') as avatar:
    avatar.create_from_letter_sequence(
        letter_images,
        ['h', 'e', 'l', 'l', 'o'],
        'outputs/hello_avatar.mp4'
    )
```

#### MediaPipe Hand Landmarks
```python
from src.models.mediapipe_extractor import MediaPipeHandExtractor

with MediaPipeHandExtractor() as extractor:
    # Extract landmarks from image
    landmarks = extractor.extract_landmarks(image)

    # Visualize landmarks
    annotated = extractor.visualize_landmarks(image)
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Data**: Image size, augmentation parameters, train/val/test splits
- **Model**: Architecture (EfficientNet, MobileNet, ResNet), layer freezing
- **Training**: Batch size, learning rate, epochs, callbacks
- **Inference**: Confidence thresholds, smoothing
- **Text-to-Sign**: Letter duration, transition effects
- **Avatar**: Canvas size, color scheme, animation settings

## 📊 Results Comparison

| Metric | Original (Flawed) | Improved (Expected) |
|--------|-------------------|---------------------|
| Test Accuracy | 99.97% | 95-98% |
| Input Size | 64×64 | 224×224 |
| Trainable Params | 21.3M (all) | ~2M (head only) |
| Data Augmentation | None | Yes |
| Validation Split | From test! ❌ | From train ✅ |
| Overfitting | Severe | Controlled |
| Generalization | Poor | Good |

## 📈 Training Tips

### For Best Results:

1. **Use Person-Based Split** (if you have person metadata)
   ```yaml
   data:
     person_based_split: true
   ```

2. **Enable Mixed Precision** (faster training on modern GPUs)
   ```yaml
   training:
     mixed_precision: true
   ```

3. **Use Early Stopping**
   ```yaml
   training:
     callbacks:
       early_stopping:
         enabled: true
         patience: 10
   ```

4. **Try Different Architectures**
   - EfficientNetV2: Best accuracy
   - MobileNetV3: Fastest inference
   - ResNet50: Good balance

5. **Fine-Tune After Initial Training**
   ```python
   # Unfreeze top layers
   for layer in model.layers[-30:]:
       layer.trainable = True

   # Train with lower learning rate
   model.compile(optimizer=Adam(lr=1e-5), ...)
   model.fit(...)
   ```

## 🎓 Research & References

See [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) for:
- State-of-the-art models (MediaPipe, YOLOv11, KD-MSLRT)
- Available datasets (FSboard, ChicagoFSWild+)
- Text-to-sign systems (GenASL, SignAvatar)
- Implementation recommendations

## 🐛 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```yaml
# Solution: Reduce batch size
training:
  batch_size: 16  # or 8
```

**Issue**: Model not improving
```yaml
# Solution: Increase learning rate or check data
training:
  learning_rate: 0.001  # Try higher
```

**Issue**: MediaPipe not detecting hands
- Ensure good lighting
- Hand should be clearly visible
- Try adjusting `min_detection_confidence`

**Issue**: Letter templates not found
```python
# Create from dataset
from src.text_to_sign.translator import create_letter_templates_from_dataset

create_letter_templates_from_dataset(
    '../ASLTransalation/fingerspelling/data',
    'data/processed/letter_templates'
)
```

## 📝 TODO / Future Enhancements

- [ ] Add dynamic signs (not just static letters)
- [ ] Implement word-level recognition
- [ ] Add ASL grammar rules for text-to-sign
- [ ] Integrate 3D avatar (Unity/Blender)
- [ ] Mobile app deployment
- [ ] Real-time webcam recognition with smoothing
- [ ] Multi-hand detection
- [ ] Sentence-level translation
- [ ] Integration with GenASL (AWS)

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Adding more sign languages
- Improving avatar realism
- Mobile deployment
- Performance optimization
- Dataset expansion

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **MediaPipe** by Google - Hand landmark detection
- **TensorFlow** team - Deep learning framework
- **ASL Dataset** contributors
- Research papers cited in RESEARCH_FINDINGS.md

## 📧 Contact

[Your Contact Information]

---

**Made with ❤️ for improving ASL accessibility**
