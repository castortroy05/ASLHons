"""
ASL Recognition & Translation Demo Application

A comprehensive Streamlit app that demonstrates:
1. Sign-to-Text: Recognize ASL fingerspelling from webcam or image
2. Text-to-Sign: Convert text to ASL fingerspelling visualization
3. Avatar Mode: Show stick figure avatar performing signs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Import our modules
from models.mediapipe_extractor import MediaPipeHandExtractor
from text_to_sign.translator import TextToSignTranslator, create_letter_templates_from_dataset
from avatar.stick_figure_avatar import StickFigureAvatar

# Page configuration
st.set_page_config(
    page_title="ASL Recognition & Translation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 1.5rem;
}
.info-box {
    background-color: #e7f3ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load trained ASL recognition model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_mediapipe():
    """Load MediaPipe hand detector."""
    return MediaPipeHandExtractor()


@st.cache_resource
def load_translator(letter_image_dir):
    """Load text-to-sign translator."""
    return TextToSignTranslator(letter_image_dir=letter_image_dir)


def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Recognition & Translation System</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🏠 Home", "📸 Sign-to-Text", "✍️ Text-to-Sign", "🤖 Avatar Demo", "ℹ️ About"]
    )

    # ===== HOME PAGE =====
    if app_mode == "🏠 Home":
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to the ASL Recognition & Translation System!

        This application provides bidirectional ASL fingerspelling translation:

        **Features:**
        - **Sign-to-Text**: Upload an image or use your webcam to recognize ASL letters
        - **Text-to-Sign**: Convert English text to ASL fingerspelling visualizations
        - **Avatar Mode**: Watch animated stick figure avatar perform signs
        - **MediaPipe Integration**: State-of-the-art hand landmark detection
        - **Deep Learning**: Powered by EfficientNetV2 with transfer learning

        **Technologies Used:**
        - TensorFlow / Keras for deep learning
        - MediaPipe for hand tracking
        - Streamlit for web interface
        - OpenCV for image processing

        **Select a mode from the sidebar to get started!** 👈
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display sample images
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📸 Sign Recognition")
            st.info("Upload images of ASL signs and get instant text translation")

        with col2:
            st.markdown("#### ✍️ Text Translation")
            st.info("Type text and see how to fingerspell it in ASL")

        with col3:
            st.markdown("#### 🤖 Avatar Demo")
            st.info("Watch an animated avatar perform ASL fingerspelling")

    # ===== SIGN-TO-TEXT PAGE =====
    elif app_mode == "📸 Sign-to-Text":
        st.markdown('<h2 class="sub-header">📸 Sign-to-Text Recognition</h2>', unsafe_allow_html=True)

        # Model selection
        model_path = st.sidebar.text_input(
            "Model Path",
            "outputs/checkpoints/improved_baseline_best.h5"
        )

        # Load model
        model = load_model(model_path)

        if model is None:
            st.error("⚠️ No model loaded. Please train a model first or check the model path.")
            st.info("To train a model, run: `python train_improved.py`")
            return

        # Input method
        input_method = st.radio("Input Method", ["Upload Image", "Webcam (Coming Soon)"])

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an ASL sign image", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Original Image")
                    st.image(image, use_column_width=True)

                with col2:
                    st.markdown("#### Hand Landmarks")

                    # Extract landmarks
                    extractor = load_mediapipe()
                    landmarks = extractor.extract_landmarks(cv2.cvtColor(image_np, cv2.COLOR_RGB2RGB))

                    if landmarks is not None and landmarks.sum() > 0:
                        # Visualize landmarks
                        annotated = extractor.visualize_landmarks(image_np.copy())
                        st.image(annotated, use_column_width=True)

                        # Predict
                        st.markdown("#### 🔮 Prediction")

                        # Preprocess image
                        img_resized = cv2.resize(image_np, (224, 224))
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        img_batch = np.expand_dims(img_normalized, axis=0)

                        # Predict
                        predictions = model.predict(img_batch, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]

                        # Get class names (you'll need to load this from your model or config)
                        class_names = list('abcdefghiklmnopqrstuvwxy')  # Excluding j and z

                        predicted_letter = class_names[predicted_class] if predicted_class < len(class_names) else "?"

                        # Display prediction
                        st.success(f"### Predicted Letter: **{predicted_letter.upper()}**")
                        st.metric("Confidence", f"{confidence * 100:.2f}%")

                        # Show top 5 predictions
                        st.markdown("#### Top 5 Predictions")
                        top5_indices = np.argsort(predictions[0])[-5:][::-1]

                        for idx in top5_indices:
                            letter = class_names[idx] if idx < len(class_names) else "?"
                            prob = predictions[0][idx]
                            st.progress(float(prob))
                            st.text(f"{letter.upper()}: {prob*100:.2f}%")

                    else:
                        st.error("⚠️ No hand detected in the image. Please upload an image with a clear hand sign.")

    # ===== TEXT-TO-SIGN PAGE =====
    elif app_mode == "✍️ Text-to-Sign":
        st.markdown('<h2 class="sub-header">✍️ Text-to-Sign Translation</h2>', unsafe_allow_html=True)

        # Letter template directory
        letter_dir = st.sidebar.text_input(
            "Letter Images Directory",
            "data/processed/letter_templates"
        )

        # Check if letter templates exist
        letter_dir_path = Path(letter_dir)
        if not letter_dir_path.exists():
            st.warning(f"⚠️ Letter template directory not found: {letter_dir}")

            if st.button("📂 Create Letter Templates from Dataset"):
                dataset_path = "../ASLTransalation/fingerspelling/data"

                with st.spinner("Creating letter templates..."):
                    try:
                        create_letter_templates_from_dataset(dataset_path, str(letter_dir_path))
                        st.success("✓ Letter templates created!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error creating templates: {e}")

            return

        # Load translator
        translator = load_translator(str(letter_dir_path))

        # Text input
        user_text = st.text_input(
            "Enter text to translate to ASL:",
            "HELLO"
        )

        if user_text:
            # Convert to letter sequence
            letter_sequence = translator.text_to_letter_sequence(user_text)

            st.info(f"**Letter Sequence:** {' '.join([l.upper() for l in letter_sequence if l != '_space_'])}")

            # Output format
            output_format = st.radio("Output Format", ["Image Grid", "Video (MP4)"])

            if st.button("🎨 Generate"):
                if output_format == "Image Grid":
                    with st.spinner("Generating image..."):
                        try:
                            output_path = f"outputs/fingerspelling_{user_text.lower().replace(' ', '_')}.png"
                            image = translator.create_fingerspelling_image(
                                user_text,
                                output_path=output_path
                            )

                            st.success("✓ Image generated!")
                            st.image(image, caption=f"ASL Fingerspelling: {user_text}")

                            # Download button
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 Download Image",
                                    f,
                                    file_name=f"asl_{user_text}.png",
                                    mime="image/png"
                                )

                        except Exception as e:
                            st.error(f"Error generating image: {e}")

                else:  # Video
                    with st.spinner("Generating video... This may take a moment."):
                        try:
                            output_path = f"outputs/fingerspelling_{user_text.lower().replace(' ', '_')}.mp4"
                            translator.create_fingerspelling_video(
                                user_text,
                                output_path=output_path
                            )

                            st.success("✓ Video generated!")

                            # Display video
                            st.video(output_path)

                            # Download button
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 Download Video",
                                    f,
                                    file_name=f"asl_{user_text}.mp4",
                                    mime="video/mp4"
                                )

                        except Exception as e:
                            st.error(f"Error generating video: {e}")

    # ===== AVATAR DEMO PAGE =====
    elif app_mode == "🤖 Avatar Demo":
        st.markdown('<h2 class="sub-header">🤖 Stick Figure Avatar Demo</h2>', unsafe_allow_html=True)

        st.info("Watch an animated stick figure avatar perform ASL fingerspelling!")

        # Settings
        col1, col2 = st.columns(2)

        with col1:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["default", "colorful", "monochrome", "dark"]
            )

        with col2:
            canvas_size = st.selectbox(
                "Canvas Size",
                [(800, 800), (1024, 1024), (1280, 720)],
                format_func=lambda x: f"{x[0]}×{x[1]}"
            )

        # Text input
        avatar_text = st.text_input(
            "Enter text for avatar to sign:",
            "HI"
        )

        # Letter directory
        letter_dir = Path("data/processed/letter_templates")

        if st.button("🎬 Generate Avatar Video"):
            if not letter_dir.exists():
                st.error("⚠️ Letter templates not found. Please create them first in Text-to-Sign mode.")
                return

            with st.spinner("Generating avatar video... This may take a minute."):
                try:
                    # Load letter images
                    import glob

                    letter_images = {}
                    for letter_path in glob.glob(str(letter_dir / "*.png")):
                        letter = Path(letter_path).stem
                        img = cv2.imread(letter_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            letter_images[letter] = img

                    if not letter_images:
                        st.error("⚠️ No letter images found in template directory.")
                        return

                    # Create avatar video
                    output_path = f"outputs/avatar_{avatar_text.lower().replace(' ', '_')}.mp4"

                    from text_to_sign.translator import TextToSignTranslator

                    translator = TextToSignTranslator()
                    letter_sequence = translator.text_to_letter_sequence(avatar_text)

                    with StickFigureAvatar(canvas_size=canvas_size, color_scheme=color_scheme) as avatar:
                        avatar.create_from_letter_sequence(
                            letter_images,
                            [l for l in letter_sequence if l != '_space_'],
                            output_path
                        )

                    st.success("✓ Avatar video generated!")

                    # Display video
                    st.video(output_path)

                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "📥 Download Avatar Video",
                            f,
                            file_name=f"avatar_{avatar_text}.mp4",
                            mime="video/mp4"
                        )

                except Exception as e:
                    st.error(f"Error generating avatar video: {e}")
                    st.exception(e)

    # ===== ABOUT PAGE =====
    elif app_mode == "ℹ️ About":
        st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)

        st.markdown("""
        ### ASL Recognition & Translation System

        **Honours Project - Improved Version**

        #### Key Improvements Over Original:

        1. **Proper Data Methodology**
           - Fixed train/validation/test split (validation from training, not test!)
           - Stratified sampling for balanced classes
           - Option for person-based split for better generalization

        2. **Modern Architecture**
           - EfficientNetV2 (better than VGG19)
           - Proper input size (224×224 instead of 64×64)
           - Frozen base layers with selective fine-tuning
           - Optimized head architecture

        3. **Data Augmentation**
           - Rotation, zoom, shift, brightness adjustments
           - NO horizontal/vertical flips (preserves ASL meaning)

        4. **MediaPipe Integration**
           - State-of-the-art hand landmark detection
           - 21 keypoints per hand
           - Can be combined with image features

        5. **Comprehensive Evaluation**
           - Confusion matrix
           - Per-class precision, recall, F1-score
           - Top-k accuracy
           - Error visualization

        6. **New Features**
           - Text-to-sign translation
           - Stick figure avatar visualization
           - Interactive web demo

        #### Technologies:
        - **TensorFlow 2.15** - Deep learning framework
        - **MediaPipe** - Hand landmark detection
        - **OpenCV** - Image/video processing
        - **Streamlit** - Web interface
        - **Scikit-learn** - Evaluation metrics

        #### Dataset:
        - 68,173 ASL fingerspelling images
        - 24 letters (A-Y, excluding J and Z)
        - Multiple signers and environments

        #### Performance:
        - Original (flawed): 99.97% (likely overfit)
        - Expected realistic: 95-98% with proper methodology

        ---

        Made with ❤️ for improving ASL accessibility
        """)


if __name__ == "__main__":
    main()
