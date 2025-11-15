"""
Text-to-Sign Translation System

Converts English text to ASL fingerspelling sequences.
Supports multiple output formats: images, videos, avatar animations.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import string


class TextToSignTranslator:
    """Translates text to ASL fingerspelling."""

    def __init__(
        self,
        letter_image_dir: Optional[str] = None,
        letter_video_dir: Optional[str] = None,
        default_letter_duration_ms: int = 800,
        transition_duration_ms: int = 200
    ):
        """
        Initialize translator.

        Args:
            letter_image_dir: Directory containing letter images (a.png, b.png, etc.)
            letter_video_dir: Directory containing letter videos (a.mp4, b.mp4, etc.)
            default_letter_duration_ms: Duration to display each letter in video
            transition_duration_ms: Duration for transitions between letters
        """
        self.letter_image_dir = Path(letter_image_dir) if letter_image_dir else None
        self.letter_video_dir = Path(letter_video_dir) if letter_video_dir else None
        self.default_letter_duration_ms = default_letter_duration_ms
        self.transition_duration_ms = transition_duration_ms

        self.letter_images = {}
        self.letter_videos = {}

        # ASL alphabet (excluding j and z which require motion)
        self.valid_letters = set(string.ascii_lowercase) - {'j', 'z'}

        # Load letter resources if directories provided
        if self.letter_image_dir and self.letter_image_dir.exists():
            self._load_letter_images()

        if self.letter_video_dir and self.letter_video_dir.exists():
            self._load_letter_videos()

    def _load_letter_images(self):
        """Load static images for each letter."""
        print(f"📂 Loading letter images from {self.letter_image_dir}...")

        for letter in self.valid_letters:
            img_path = self.letter_image_dir / f"{letter}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.letter_images[letter] = img

        print(f"   ✓ Loaded {len(self.letter_images)} letter images")

    def _load_letter_videos(self):
        """Load video clips for each letter."""
        print(f"📂 Loading letter videos from {self.letter_video_dir}...")

        for letter in self.valid_letters:
            for ext in ['.mp4', '.avi', '.mov']:
                video_path = self.letter_video_dir / f"{letter}{ext}"
                if video_path.exists():
                    self.letter_videos[letter] = str(video_path)
                    break

        print(f"   ✓ Loaded {len(self.letter_videos)} letter videos")

    def text_to_letter_sequence(self, text: str) -> List[str]:
        """
        Convert text to sequence of ASL letters.

        Args:
            text: Input text

        Returns:
            List of letters to sign
        """
        # Convert to lowercase and filter to valid letters
        letters = []
        for char in text.lower():
            if char in self.valid_letters:
                letters.append(char)
            elif char == ' ':
                letters.append('_space_')  # Placeholder for space

        return letters

    def create_fingerspelling_image(
        self,
        text: str,
        output_path: Optional[str] = None,
        image_size: Tuple[int, int] = (128, 128),
        max_letters_per_row: int = 10,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Create a composite image showing fingerspelling of text.

        Args:
            text: Text to fingerspell
            output_path: Path to save image (optional)
            image_size: Size of each letter image
            max_letters_per_row: Maximum letters per row
            background_color: Background color (RGB)

        Returns:
            Composite image
        """
        if not self.letter_images:
            raise ValueError("No letter images loaded. Provide letter_image_dir when initializing.")

        letters = self.text_to_letter_sequence(text)

        # Calculate grid dimensions
        n_letters = len(letters)
        n_cols = min(n_letters, max_letters_per_row)
        n_rows = (n_letters + n_cols - 1) // n_cols

        # Create canvas
        canvas_height = n_rows * image_size[1] + (n_rows + 1) * 20  # 20px padding
        canvas_width = n_cols * image_size[0] + (n_cols + 1) * 20
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

        # Place letters
        for i, letter in enumerate(letters):
            if letter == '_space_':
                continue  # Skip spaces

            if letter not in self.letter_images:
                print(f"⚠️  Warning: No image for letter '{letter}'")
                continue

            row = i // n_cols
            col = i % n_cols

            # Get letter image
            img = self.letter_images[letter]
            img_resized = cv2.resize(img, image_size)

            # Calculate position
            y = row * (image_size[1] + 20) + 20
            x = col * (image_size[0] + 20) + 20

            # Place on canvas
            canvas[y:y+image_size[1], x:x+image_size[0]] = img_resized

            # Add letter label
            cv2.putText(
                canvas,
                letter.upper(),
                (x + image_size[0]//2 - 10, y + image_size[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print(f"✓ Fingerspelling image saved to {output_path}")

        return canvas

    def create_fingerspelling_video(
        self,
        text: str,
        output_path: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (1280, 720),
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        """
        Create a video showing fingerspelling of text.

        Args:
            text: Text to fingerspell
            output_path: Path to save video
            fps: Frames per second
            resolution: Video resolution (width, height)
            background_color: Background color (RGB)
        """
        if not self.letter_images:
            raise ValueError("No letter images loaded. Provide letter_image_dir when initializing.")

        letters = self.text_to_letter_sequence(text)

        # Setup video writer
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            resolution
        )

        # Calculate frames per letter
        frames_per_letter = int((self.default_letter_duration_ms / 1000.0) * fps)
        transition_frames = int((self.transition_duration_ms / 1000.0) * fps)

        print(f"🎬 Creating fingerspelling video for: '{text}'")
        print(f"   Letters: {len(letters)}, FPS: {fps}, Duration: ~{len(letters) * self.default_letter_duration_ms / 1000:.1f}s")

        for i, letter in enumerate(letters):
            if letter == '_space_':
                # Show blank frame for space
                blank = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
                for _ in range(frames_per_letter // 2):
                    out.write(cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))
                continue

            if letter not in self.letter_images:
                print(f"⚠️  Warning: No image for letter '{letter}'")
                continue

            # Get letter image
            img = self.letter_images[letter]

            # Create frame with letter centered
            frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)

            # Calculate scaling to fit in frame
            scale = min(resolution[0] * 0.6 / img.shape[1], resolution[1] * 0.6 / img.shape[0])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img_resized = cv2.resize(img, new_size)

            # Center image
            y_offset = (resolution[1] - img_resized.shape[0]) // 2
            x_offset = (resolution[0] - img_resized.shape[1]) // 2

            frame[y_offset:y_offset+img_resized.shape[0], x_offset:x_offset+img_resized.shape[1]] = img_resized

            # Add text overlay
            text_overlay = f"Letter: {letter.upper()} ({i+1}/{len(letters)})"
            cv2.putText(
                frame,
                text_overlay,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                3
            )

            # Full text being spelled
            spelled_so_far = ''.join([l for l in letters[:i+1] if l != '_space_'])
            cv2.putText(
                frame,
                spelled_so_far.upper(),
                (50, resolution[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 100, 200),
                4
            )

            # Write frames for this letter
            for _ in range(frames_per_letter):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Transition to next letter (simple fade)
            if i < len(letters) - 1 and transition_frames > 0:
                for t in range(transition_frames):
                    alpha = t / transition_frames
                    faded = (frame * (1 - alpha)).astype(np.uint8)
                    out.write(cv2.cvtColor(faded, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"✓ Fingerspelling video saved to {output_path}")

    def translate(
        self,
        text: str,
        output_format: str = 'sequence',
        output_path: Optional[str] = None,
        **kwargs
    ):
        """
        Translate text to ASL.

        Args:
            text: Input text
            output_format: 'sequence', 'image', or 'video'
            output_path: Output path (for image/video)
            **kwargs: Additional arguments for specific output formats

        Returns:
            Letter sequence (if format='sequence') or path to output file
        """
        if output_format == 'sequence':
            return self.text_to_letter_sequence(text)

        elif output_format == 'image':
            if not output_path:
                output_path = 'outputs/fingerspelling_image.png'
            return self.create_fingerspelling_image(text, output_path, **kwargs)

        elif output_format == 'video':
            if not output_path:
                output_path = 'outputs/fingerspelling_video.mp4'
            self.create_fingerspelling_video(text, output_path, **kwargs)
            return output_path

        else:
            raise ValueError(f"Unknown output format: {output_format}")


def create_letter_templates_from_dataset(
    dataset_path: str,
    output_dir: str,
    samples_per_letter: int = 5
):
    """
    Create letter template images from the dataset.

    Args:
        dataset_path: Path to ASL dataset
        output_dir: Directory to save letter templates
        samples_per_letter: Number of samples to save per letter
    """
    import glob

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Creating letter templates from {dataset_path}...")

    for letter_dir in glob.glob(os.path.join(dataset_path, '*')):
        if not os.path.isdir(letter_dir):
            continue

        letter = os.path.basename(letter_dir)
        images = glob.glob(os.path.join(letter_dir, '*.png'))

        if not images:
            continue

        # Select representative images
        selected = np.random.choice(images, min(samples_per_letter, len(images)), replace=False)

        # Save first one as the main template
        img = cv2.imread(selected[0])
        if img is not None:
            # Resize to standard size
            img_resized = cv2.resize(img, (256, 256))
            cv2.imwrite(str(output_path / f"{letter}.png"), img_resized)

        print(f"   ✓ {letter}: template created")

    print(f"✓ Letter templates saved to {output_dir}")
