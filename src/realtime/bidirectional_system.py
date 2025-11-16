"""
Real-Time Bidirectional Sign Language Communication System

Enables live conversations between:
- Deaf person (signing) ↔ Hearing person (speaking)
- Deaf person (signing) ↔ Deaf person (signing, different language)
- Real-time captioning for sign language videos
"""

import cv2
import numpy as np
from threading import Thread
from queue import Queue
from collections import deque
import time
from typing import Optional, Dict, Callable
import logging


class BidirectionalSignSystem:
    """
    Real-time bidirectional sign language communication

    Features:
    - Sign → Text → Speech (for hearing users)
    - Speech → Text → Sign (for deaf users)
    - Sign → Sign (translation between sign languages)
    - Real-time captioning
    """

    def __init__(
        self,
        sign_language: str = 'ASL',
        spoken_language: str = 'en-US',
        mode: str = 'bidirectional'
    ):
        """
        Initialize bidirectional system

        Args:
            sign_language: Sign language code (ASL, BSL, etc.)
            spoken_language: Spoken language code
            mode: 'bidirectional', 'deaf_to_hearing', 'hearing_to_deaf'
        """
        self.sign_language = sign_language
        self.spoken_language = spoken_language
        self.mode = mode

        # Import components (lazy loading)
        self._setup_components()

        # Processing queues
        self.video_queue = Queue(maxsize=100)
        self.audio_queue = Queue(maxsize=100)
        self.caption_queue = Queue(maxsize=50)
        self.avatar_queue = Queue(maxsize=50)

        # State
        self.is_running = False
        self.threads = []

        logging.info(f"Initialized {mode} system for {sign_language} ↔ {spoken_language}")

    def _setup_components(self):
        """Lazy load all required components"""
        # Sign recognition
        from models.temporal_recognition import RealtimeSignRecognizer
        from models.mediapipe_extractor import MediaPipeHandExtractor

        self.sign_recognizer = RealtimeSignRecognizer(
            model=None,  # Load from checkpoint
            landmark_extractor=MediaPipeHandExtractor(),
            vocabulary=[],  # Load language-specific
            language=self.sign_language
        )

        # Speech recognition (Whisper)
        try:
            import whisper
            self.speech_recognizer = whisper.load_model("base")
        except ImportError:
            logging.warning("Whisper not available. Install: pip install openai-whisper")
            self.speech_recognizer = None

        # Text-to-speech
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
        except ImportError:
            logging.warning("pyttsx3 not available. Install: pip install pyttsx3")
            self.tts_engine = None

        # Sign generation
        from text_to_sign.translator import TextToSignTranslator
        from avatar.stick_figure_avatar import StickFigureAvatar

        self.sign_generator = TextToSignTranslator()
        self.avatar = StickFigureAvatar()

        # Language manager
        from utils.language_manager import SignLanguageManager, GrammarTransformer

        self.language_manager = SignLanguageManager()
        self.grammar_transformer = GrammarTransformer(self.sign_language)

    def start(self):
        """Start the bidirectional communication system"""
        if self.is_running:
            logging.warning("System already running")
            return

        self.is_running = True

        # Start processing threads
        if self.mode in ['bidirectional', 'deaf_to_hearing']:
            # Video processing thread (sign recognition)
            video_thread = Thread(target=self._process_video_stream)
            video_thread.daemon = True
            video_thread.start()
            self.threads.append(video_thread)

        if self.mode in ['bidirectional', 'hearing_to_deaf']:
            # Audio processing thread (speech recognition)
            audio_thread = Thread(target=self._process_audio_stream)
            audio_thread.daemon = True
            audio_thread.start()
            self.threads.append(audio_thread)

        # Display thread
        display_thread = Thread(target=self._update_display)
        display_thread.daemon = True
        display_thread.start()
        self.threads.append(display_thread)

        logging.info("Bidirectional system started")

    def stop(self):
        """Stop the system"""
        self.is_running = False

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)

        self.threads = []
        logging.info("Bidirectional system stopped")

    def _process_video_stream(self):
        """
        Process video stream for sign recognition

        Pipeline:
        1. Capture video frames
        2. Extract landmarks
        3. Recognize signs
        4. Convert to text
        5. Display captions
        6. Optional: speak out loud (TTS)
        """
        cap = cv2.VideoCapture(0)  # Default camera

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Recognize signs
            result = self.sign_recognizer.process_frame(frame)

            if result:
                signs = result['signs']
                confidence = result['confidence']

                # Convert to text
                text = ' '.join(signs)

                # Add to caption queue
                self.caption_queue.put({
                    'text': text,
                    'confidence': confidence,
                    'type': 'sign_recognition',
                    'timestamp': time.time()
                })

                # Optional: speak out loud
                if self.mode == 'deaf_to_hearing' and self.tts_engine and confidence > 0.7:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()

        cap.release()

    def _process_audio_stream(self):
        """
        Process audio stream for speech recognition

        Pipeline:
        1. Capture audio
        2. Recognize speech (Whisper)
        3. Transform to sign language grammar
        4. Generate sign sequence
        5. Display avatar
        """
        import pyaudio

        # Audio settings
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        audio_buffer = []
        buffer_duration = 3  # seconds

        while self.is_running:
            # Capture audio chunks
            data = stream.read(CHUNK)
            audio_buffer.append(np.frombuffer(data, dtype=np.int16))

            # Process every 3 seconds
            if len(audio_buffer) >= buffer_duration * (RATE // CHUNK):
                # Convert to audio array
                audio_array = np.concatenate(audio_buffer).astype(np.float32) / 32768.0

                # Recognize speech
                if self.speech_recognizer:
                    result = self.speech_recognizer.transcribe(audio_array)
                    text = result['text']

                    # Transform to sign language grammar
                    sign_gloss = self.grammar_transformer.transform(text)

                    # Generate signs
                    sign_sequence = self.grammar_transformer.add_nonmanual_markers(sign_gloss)

                    # Add to avatar queue
                    self.avatar_queue.put({
                        'text': text,
                        'sign_gloss': sign_gloss,
                        'signs': sign_sequence,
                        'timestamp': time.time()
                    })

                # Clear buffer
                audio_buffer = []

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _update_display(self):
        """
        Update display with captions and avatar

        Layout:
        - Top: Sign recognition captions
        - Center: Video feed + Avatar
        - Bottom: Speech recognition text
        """
        import cv2

        # Create display window
        window_width = 1280
        window_height = 720

        while self.is_running:
            # Create blank canvas
            display = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255

            # Display captions from sign recognition
            if not self.caption_queue.empty():
                caption_data = self.caption_queue.get()
                text = caption_data['text']
                confidence = caption_data['confidence']

                # Draw caption at top
                cv2.putText(
                    display,
                    f"Sign: {text} ({confidence:.0%})",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 0),
                    2
                )

            # Display avatar from speech recognition
            if not self.avatar_queue.empty():
                avatar_data = self.avatar_queue.get()
                text = avatar_data['text']
                sign_gloss = avatar_data['sign_gloss']

                # Draw at bottom
                cv2.putText(
                    display,
                    f"Speech: {text}",
                    (20, window_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 100, 200),
                    2
                )

                cv2.putText(
                    display,
                    f"Signs: {sign_gloss}",
                    (20, window_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 150, 0),
                    2
                )

            # Show display
            cv2.imshow('Sign Language Communication', display)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

            time.sleep(0.033)  # ~30 FPS

        cv2.destroyAllWindows()


class RealtimeCaptioning:
    """
    Real-time captioning system for sign language

    Similar to YouTube auto-captions but for sign language
    """

    def __init__(
        self,
        sign_language: str = 'ASL',
        confidence_threshold: float = 0.6
    ):
        self.sign_language = sign_language
        self.confidence_threshold = confidence_threshold

        # Sign recognizer
        from models.temporal_recognition import RealtimeSignRecognizer
        from models.mediapipe_extractor import MediaPipeHandExtractor

        self.recognizer = RealtimeSignRecognizer(
            model=None,
            landmark_extractor=MediaPipeHandExtractor(),
            vocabulary=[],
            language=sign_language
        )

        # Caption buffer
        self.caption_buffer = deque(maxlen=10)
        self.current_caption = ""

    def process_video_file(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ):
        """
        Generate captions for sign language video file

        Args:
            video_path: Path to input video
            output_path: Path to save captioned video
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recognize signs
            result = self.recognizer.process_frame(frame)

            if result and result['confidence'] > self.confidence_threshold:
                caption_text = ' '.join(result['signs'])
                self.current_caption = caption_text

                # Add to buffer
                self.caption_buffer.append({
                    'text': caption_text,
                    'confidence': result['confidence'],
                    'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                })

            # Draw caption on frame
            if self.current_caption:
                self._draw_caption(frame, self.current_caption)

            # Write frame
            if writer:
                writer.write(frame)

            # Display
            cv2.imshow('Sign Language Captioning', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        return list(self.caption_buffer)

    def _draw_caption(self, frame, text, position='bottom'):
        """
        Draw caption on video frame

        Args:
            frame: Video frame
            text: Caption text
            position: 'top' or 'bottom'
        """
        height, width = frame.shape[:2]

        # Create caption box
        box_height = 80
        if position == 'bottom':
            y_start = height - box_height
            y_end = height
        else:
            y_start = 0
            y_end = box_height

        # Semi-transparent black box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (width, y_end), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)

        # Center text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = y_start + (box_height + text_height) // 2

        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


# Example usage
if __name__ == "__main__":
    # Example 1: Bidirectional communication
    system = BidirectionalSignSystem(
        sign_language='ASL',
        spoken_language='en-US',
        mode='bidirectional'
    )

    print("Starting bidirectional sign language communication...")
    print("Press 'q' to quit")

    system.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        system.stop()

    # Example 2: Real-time captioning
    # captioner = RealtimeCaptioning(sign_language='ASL')
    # captions = captioner.process_video_file('input_video.mp4', 'captioned_output.mp4')
