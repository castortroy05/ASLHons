"""
MediaPipe-based hand landmark extraction for ASL recognition.

MediaPipe provides state-of-the-art hand tracking with 21 keypoints per hand.
This can significantly improve recognition accuracy by focusing on hand structure.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
from tqdm import tqdm


class MediaPipeHandExtractor:
    """Extract hand landmarks using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe hand detector.

        Args:
            static_image_mode: Whether to treat each image independently
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmark_features_size = 21 * 3  # 21 keypoints × (x, y, z)

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.

        Args:
            image: Input image (RGB, values in [0, 1] or [0, 255])

        Returns:
            Flattened landmark array of shape (63,) or None if no hand detected
            Contains [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
        """
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Process image
        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            # Get first (dominant) hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            return np.array(landmarks, dtype=np.float32)
        else:
            # No hand detected - return zeros
            return np.zeros(self.landmark_features_size, dtype=np.float32)

    def extract_landmarks_batch(
        self,
        images: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract landmarks from batch of images.

        Args:
            images: Batch of images (N, H, W, C)
            show_progress: Whether to show progress bar

        Returns:
            Landmark features array of shape (N, 63)
        """
        landmarks_batch = []

        iterator = tqdm(images, desc="Extracting landmarks") if show_progress else images

        for image in iterator:
            landmarks = self.extract_landmarks(image)
            landmarks_batch.append(landmarks)

        return np.array(landmarks_batch, dtype=np.float32)

    def visualize_landmarks(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize hand landmarks on image.

        Args:
            image: Input image
            landmarks: Optional pre-extracted landmarks (63,)

        Returns:
            Image with landmarks drawn
        """
        # Make a copy
        annotated_image = image.copy()

        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                annotated_image = (annotated_image * 255).astype(np.uint8)

        # Get landmarks if not provided
        if landmarks is None:
            results = self.hands.process(annotated_image)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
            else:
                return annotated_image
        else:
            # Convert flat landmarks back to MediaPipe format
            # This is a simplified version - for full visualization,
            # we need to reconstruct the MediaPipe landmark object
            return annotated_image

        # Draw landmarks
        if 'hand_landmarks' in locals():
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        return annotated_image

    def get_landmark_distances(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between landmarks (geometric features).

        Args:
            landmarks: Flat landmark array (63,)

        Returns:
            Distance features (useful for rotation/translation invariance)
        """
        # Reshape to (21, 3)
        points = landmarks.reshape(21, 3)

        # Compute distances from wrist (landmark 0) to all other landmarks
        wrist = points[0]
        distances = np.linalg.norm(points - wrist, axis=1)

        # Also compute finger lengths
        # Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
        finger_lengths = []
        finger_indices = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16],# Ring
            [17, 18, 19, 20] # Pinky
        ]

        for finger in finger_indices:
            length = 0
            for i in range(len(finger) - 1):
                length += np.linalg.norm(points[finger[i+1]] - points[finger[i]])
            finger_lengths.append(length)

        # Combine features
        features = np.concatenate([distances, finger_lengths])

        return features

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def augment_with_landmarks(
    images: np.ndarray,
    extractor: Optional[MediaPipeHandExtractor] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract landmarks for all images in dataset.

    Args:
        images: Image array (N, H, W, C)
        extractor: Optional MediaPipe extractor (creates one if None)
        show_progress: Whether to show progress

    Returns:
        Tuple of (images, landmarks)
    """
    if extractor is None:
        extractor = MediaPipeHandExtractor()
        close_after = True
    else:
        close_after = False

    try:
        landmarks = extractor.extract_landmarks_batch(images, show_progress=show_progress)
        return images, landmarks
    finally:
        if close_after:
            extractor.close()


def preprocess_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize and preprocess landmarks.

    Args:
        landmarks: Raw landmarks (N, 63)

    Returns:
        Preprocessed landmarks
    """
    # Center landmarks (subtract mean)
    centered = landmarks - landmarks.mean(axis=0)

    # Scale to unit variance
    std = landmarks.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    normalized = centered / std

    return normalized
