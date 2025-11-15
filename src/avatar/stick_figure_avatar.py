"""
Stick Figure Avatar for ASL Visualization

Creates animated stick figure avatars from MediaPipe hand landmarks.
This provides a simple, effective way to visualize sign language without
needing complex 3D models.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional, Dict
from pathlib import Path


class StickFigureAvatar:
    """Create stick figure avatar from hand landmarks."""

    # MediaPipe hand connections (pairs of landmark indices)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)            # Palm
    ]

    # Color schemes
    COLOR_SCHEMES = {
        'default': {
            'hand_connections': (0, 255, 0),    # Green
            'hand_points': (0, 0, 255),         # Red
            'background': (255, 255, 255),      # White
        },
        'colorful': {
            'hand_connections': (255, 140, 0),  # Orange
            'hand_points': (147, 20, 255),      # Purple
            'background': (240, 248, 255),      # Alice blue
        },
        'monochrome': {
            'hand_connections': (50, 50, 50),   # Dark gray
            'hand_points': (0, 0, 0),           # Black
            'background': (255, 255, 255),      # White
        },
        'dark': {
            'hand_connections': (0, 255, 0),    # Green
            'hand_points': (0, 200, 255),       # Yellow
            'background': (30, 30, 30),         # Dark gray
        }
    }

    def __init__(
        self,
        canvas_size: Tuple[int, int] = (800, 800),
        line_thickness: int = 3,
        point_radius: int = 5,
        color_scheme: str = 'default'
    ):
        """
        Initialize stick figure avatar.

        Args:
            canvas_size: Size of output canvas (width, height)
            line_thickness: Thickness of connection lines
            point_radius: Radius of landmark points
            color_scheme: Color scheme name
        """
        self.canvas_size = canvas_size
        self.line_thickness = line_thickness
        self.point_radius = point_radius

        if color_scheme not in self.COLOR_SCHEMES:
            print(f"⚠️  Unknown color scheme: {color_scheme}. Using 'default'")
            color_scheme = 'default'

        self.colors = self.COLOR_SCHEMES[color_scheme]

        # Initialize MediaPipe for extracting landmarks from images
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def landmarks_to_coordinates(
        self,
        landmarks: np.ndarray,
        scale: float = 0.8
    ) -> np.ndarray:
        """
        Convert normalized landmarks to canvas coordinates.

        Args:
            landmarks: Flat array of landmarks (63,) or structured (21, 3)
            scale: Scale factor for hand size

        Returns:
            Array of (x, y) coordinates (21, 2)
        """
        if landmarks.shape == (63,):
            landmarks = landmarks.reshape(21, 3)

        # Extract x, y coordinates (ignore z for 2D visualization)
        coords_normalized = landmarks[:, :2]

        # Center and scale to canvas
        center_x, center_y = self.canvas_size[0] // 2, self.canvas_size[1] // 2

        # Scale to canvas size
        scale_factor = min(self.canvas_size) * scale

        coords = coords_normalized.copy()

        # Center the hand
        mean_x, mean_y = coords[:, 0].mean(), coords[:, 1].mean()
        coords[:, 0] -= mean_x
        coords[:, 1] -= mean_y

        # Scale
        coords *= scale_factor

        # Flip Y axis (image coordinates have origin at top-left)
        coords[:, 1] *= -1

        # Translate to center of canvas
        coords[:, 0] += center_x
        coords[:, 1] += center_y

        return coords.astype(int)

    def draw_hand(
        self,
        landmarks: np.ndarray,
        canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draw hand on canvas.

        Args:
            landmarks: Hand landmarks (63,) or (21, 3)
            canvas: Optional existing canvas (creates new if None)

        Returns:
            Canvas with hand drawn
        """
        if canvas is None:
            canvas = np.ones(
                (self.canvas_size[1], self.canvas_size[0], 3),
                dtype=np.uint8
            ) * np.array(self.colors['background'], dtype=np.uint8)

        # Convert landmarks to coordinates
        coords = self.landmarks_to_coordinates(landmarks)

        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = tuple(coords[start_idx])
            end_point = tuple(coords[end_idx])

            cv2.line(
                canvas,
                start_point,
                end_point,
                self.colors['hand_connections'],
                self.line_thickness
            )

        # Draw points
        for coord in coords:
            cv2.circle(
                canvas,
                tuple(coord),
                self.point_radius,
                self.colors['hand_points'],
                -1
            )

        return canvas

    def extract_and_draw(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract landmarks from image and draw stick figure.

        Args:
            image: Input image (RGB)

        Returns:
            Canvas with stick figure or None if no hand detected
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

        # Process image
        results = self.hands.process(image)

        if not results.multi_hand_landmarks:
            return None

        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # Draw stick figure
        return self.draw_hand(landmarks)

    def create_video_from_images(
        self,
        images: List[np.ndarray],
        output_path: str,
        fps: int = 30,
        frames_per_image: int = 30,
        add_transitions: bool = True
    ):
        """
        Create video of stick figures from sequence of images.

        Args:
            images: List of input images
            output_path: Path to save video
            fps: Frames per second
            frames_per_image: Number of frames to display each image
            add_transitions: Whether to add smooth transitions
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            self.canvas_size
        )

        print(f"🎬 Creating stick figure video...")
        print(f"   Images: {len(images)}, FPS: {fps}")

        landmarks_sequence = []

        # Extract landmarks from all images
        for i, img in enumerate(images):
            results = self.hands.process(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8))

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks_sequence.append(np.array(landmarks, dtype=np.float32))
            else:
                print(f"⚠️  Warning: No hand detected in image {i}")
                # Use previous landmarks if available
                if landmarks_sequence:
                    landmarks_sequence.append(landmarks_sequence[-1])

        # Generate frames
        for i, landmarks in enumerate(landmarks_sequence):
            # Draw main frame
            frame = self.draw_hand(landmarks)

            # Add label
            cv2.putText(
                frame,
                f"Frame {i+1}/{len(landmarks_sequence)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2
            )

            # Write main frames
            for _ in range(frames_per_image):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Add transition
            if add_transitions and i < len(landmarks_sequence) - 1:
                # Interpolate between current and next landmarks
                next_landmarks = landmarks_sequence[i + 1]
                transition_steps = fps // 6  # ~0.17 seconds transition

                for t in range(transition_steps):
                    alpha = t / transition_steps
                    interpolated = landmarks * (1 - alpha) + next_landmarks * alpha

                    transition_frame = self.draw_hand(interpolated)
                    out.write(cv2.cvtColor(transition_frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"✓ Stick figure video saved to {output_path}")

    def create_from_letter_sequence(
        self,
        letter_images: Dict[str, np.ndarray],
        letter_sequence: List[str],
        output_path: str,
        fps: int = 30,
        frames_per_letter: int = 24
    ):
        """
        Create stick figure video from letter sequence.

        Args:
            letter_images: Dictionary mapping letters to images
            letter_sequence: Sequence of letters to display
            output_path: Path to save video
            fps: Frames per second
            frames_per_letter: Frames per letter
        """
        images = []
        for letter in letter_sequence:
            if letter in letter_images:
                images.append(letter_images[letter])
            else:
                print(f"⚠️  Warning: No image for letter '{letter}'")

        if images:
            self.create_video_from_images(
                images,
                output_path,
                fps=fps,
                frames_per_image=frames_per_letter
            )
        else:
            print("✗ No valid images to create video")

    def close(self):
        """Release resources."""
        self.hands.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_avatar_from_text(
    text: str,
    letter_images: Dict[str, np.ndarray],
    output_path: str,
    canvas_size: Tuple[int, int] = (800, 800),
    color_scheme: str = 'default'
):
    """
    Create stick figure avatar video from text.

    Args:
        text: Text to convert to sign language
        letter_images: Dictionary of letter images
        output_path: Path to save video
        canvas_size: Canvas size
        color_scheme: Color scheme
    """
    from text_to_sign.translator import TextToSignTranslator

    # Convert text to letter sequence
    translator = TextToSignTranslator()
    letter_sequence = translator.text_to_letter_sequence(text)

    # Create avatar
    with StickFigureAvatar(canvas_size=canvas_size, color_scheme=color_scheme) as avatar:
        avatar.create_from_letter_sequence(
            letter_images,
            letter_sequence,
            output_path
        )
