"""
Data augmentation for ASL recognition.

CRITICAL CONSIDERATIONS FOR SIGN LANGUAGE:
1. NO horizontal flip - ASL is not symmetric!
2. NO vertical flip - would completely change meaning
3. Moderate rotation only - extreme rotations are unrealistic
4. Preserve hand structure and orientation
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Any


class ASLAugmentation:
    """Data augmentation pipeline for ASL images."""

    def __init__(
        self,
        rotation_range: float = 15,
        width_shift_range: float = 0.1,
        height_shift_range: float = 0.1,
        zoom_range: float = 0.15,
        brightness_range: tuple = (0.8, 1.2),
        fill_mode: str = "nearest",
        horizontal_flip: bool = False,  # NEVER flip for ASL!
        vertical_flip: bool = False,    # NEVER flip for ASL!
    ):
        """
        Initialize augmentation pipeline.

        Args:
            rotation_range: Rotation angle range in degrees
            width_shift_range: Horizontal shift as fraction of width
            height_shift_range: Vertical shift as fraction of height
            zoom_range: Zoom range (0.15 = 85% to 115%)
            brightness_range: Brightness adjustment range
            fill_mode: Fill mode for pixels outside boundaries
            horizontal_flip: Whether to allow horizontal flipping (should be False!)
            vertical_flip: Whether to allow vertical flipping (should be False!)
        """
        if horizontal_flip:
            print("⚠️  WARNING: Horizontal flip enabled - this may harm ASL recognition!")
        if vertical_flip:
            print("⚠️  WARNING: Vertical flip enabled - this may harm ASL recognition!")

        self.augmentation_params = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'zoom_range': zoom_range,
            'brightness_range': brightness_range,
            'fill_mode': fill_mode,
            'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip,
        }

        self.train_generator = None
        self.val_generator = None

    def get_train_generator(self) -> ImageDataGenerator:
        """
        Get training data generator with augmentation.

        Returns:
            ImageDataGenerator for training
        """
        if self.train_generator is None:
            self.train_generator = ImageDataGenerator(**self.augmentation_params)

        return self.train_generator

    def get_val_generator(self) -> ImageDataGenerator:
        """
        Get validation data generator (no augmentation, only rescaling).

        Returns:
            ImageDataGenerator for validation/testing
        """
        if self.val_generator is None:
            # No augmentation for validation/test
            self.val_generator = ImageDataGenerator()

        return self.val_generator

    def flow(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = True
    ):
        """
        Create data flow iterator.

        Args:
            X: Input images
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation

        Returns:
            Data generator iterator
        """
        generator = self.get_train_generator() if augment else self.get_val_generator()

        return generator.flow(
            X, y,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def augment_batch(self, images: np.ndarray, n_augmentations: int = 5) -> np.ndarray:
        """
        Apply augmentation to a batch of images.

        Args:
            images: Input images (batch_size, height, width, channels)
            n_augmentations: Number of augmented versions per image

        Returns:
            Augmented images
        """
        generator = self.get_train_generator()
        augmented_images = []

        for img in images:
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)

            # Generate augmented versions
            aug_iter = generator.flow(img_batch, batch_size=1)

            for _ in range(n_augmentations):
                augmented_images.append(next(aug_iter)[0])

        return np.array(augmented_images)

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> 'ASLAugmentation':
        """
        Create augmentation from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ASLAugmentation instance
        """
        aug_config = config.get('data', {}).get('augmentation', {})

        if not aug_config.get('enabled', True):
            print("⚠️  Data augmentation is DISABLED in config")
            # Return augmentation with no transformations
            return ASLAugmentation(
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                zoom_range=0,
                brightness_range=(1.0, 1.0),
            )

        return ASLAugmentation(
            rotation_range=aug_config.get('rotation_range', 15),
            width_shift_range=aug_config.get('width_shift_range', 0.1),
            height_shift_range=aug_config.get('height_shift_range', 0.1),
            zoom_range=aug_config.get('zoom_range', 0.15),
            brightness_range=tuple(aug_config.get('brightness_range', [0.8, 1.2])),
            fill_mode=aug_config.get('fill_mode', 'nearest'),
            horizontal_flip=aug_config.get('horizontal_flip', False),
            vertical_flip=aug_config.get('vertical_flip', False),
        )


def preview_augmentations(
    images: np.ndarray,
    labels: np.ndarray,
    augmentation: ASLAugmentation,
    n_samples: int = 5,
    n_augmentations: int = 5
):
    """
    Preview augmentation effects.

    Args:
        images: Sample images
        labels: Sample labels
        augmentation: Augmentation pipeline
        n_samples: Number of original images to show
        n_augmentations: Number of augmented versions per image
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_samples, n_augmentations + 1, figsize=(15, 3 * n_samples))

    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Original\nLabel: {labels[i].argmax()}")
        axes[i, 0].axis('off')

        # Augmented images
        img_batch = np.expand_dims(images[i], axis=0)
        aug_gen = augmentation.get_train_generator().flow(img_batch, batch_size=1)

        for j in range(n_augmentations):
            aug_img = next(aug_gen)[0]
            axes[i, j + 1].imshow(aug_img)
            axes[i, j + 1].set_title(f"Augmented {j+1}")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/augmentation_preview.png', dpi=150, bbox_inches='tight')
    print("✓ Augmentation preview saved to outputs/augmentation_preview.png")
    plt.show()
