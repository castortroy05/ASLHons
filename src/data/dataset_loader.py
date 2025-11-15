"""
Dataset loader with proper train/validation/test splits.
Addresses critical issues from original implementation:
1. Proper validation set (from training data, not test data!)
2. Stratification to handle class imbalance
3. Person-based split option for true generalization
4. No data leakage
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


class ASLDatasetLoader:
    """Loads ASL fingerspelling dataset with proper data splits."""

    def __init__(
        self,
        dataset_path: str,
        image_size: int = 224,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
        person_based_split: bool = False
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size (square)
            test_size: Fraction of data for test set (0.0-1.0)
            val_size: Fraction of TRAINING data for validation set
            stratify: Whether to stratify splits by class
            random_state: Random seed for reproducibility
            person_based_split: If True, split by person (requires person metadata)
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state
        self.person_based_split = person_based_split

        self.label_binarizer = LabelBinarizer()
        self.classes = None
        self.class_distribution = None

    def load_dataset_info(self) -> pd.DataFrame:
        """
        Load dataset information without loading images.

        Returns:
            DataFrame with columns: ['image_path', 'label', 'person_id']
        """
        print("📂 Loading dataset information...")

        data_dict = {}
        for directory in glob.glob(os.path.join(self.dataset_path, '*')):
            if os.path.isdir(directory):
                label = os.path.basename(directory)
                images = glob.glob(os.path.join(directory, '*.png'))

                if images:
                    data_dict[label] = images
                    print(f"   {label}: {len(images)} images")

        # Create DataFrame
        rows = []
        for label, image_paths in data_dict.items():
            for img_path in image_paths:
                # Extract person_id from filename if available
                # Assuming format: person_X_gesture_Y.png
                filename = os.path.basename(img_path)
                person_id = self._extract_person_id(filename)

                rows.append({
                    'image_path': img_path,
                    'label': label,
                    'person_id': person_id
                })

        df = pd.DataFrame(rows)

        # Store class information
        self.classes = sorted(df['label'].unique())
        self.class_distribution = df['label'].value_counts().to_dict()

        print(f"\n✓ Dataset loaded: {len(df)} images, {len(self.classes)} classes")
        print(f"  Classes: {self.classes}")

        return df

    def _extract_person_id(self, filename: str) -> Optional[int]:
        """
        Extract person ID from filename if available.

        Args:
            filename: Image filename

        Returns:
            Person ID or None
        """
        # Try to extract person ID from filename
        # This is dataset-specific - adjust regex as needed
        import re

        # Common patterns:
        # person_X_..., pX_..., subject_X_...
        patterns = [
            r'person_(\d+)',
            r'subject_(\d+)',
            r'p(\d+)_',
            r'user(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def create_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with proper methodology.

        CRITICAL: This fixes the major issue in the original code where
        validation set was created from test set!

        Args:
            df: DataFrame with dataset information

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n🔀 Creating train/validation/test splits...")

        if self.person_based_split and df['person_id'].notna().any():
            print("   Using person-based split (better generalization)")
            return self._person_based_split(df)
        else:
            print("   Using random stratified split")
            return self._random_split(df)

    def _random_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random stratified split.

        Strategy:
        1. Split into train_temp (80%) and test (20%)
        2. Split train_temp into train (90%) and val (10%)
        Result: train ~72%, val ~8%, test ~20%
        """
        stratify_col = df['label'] if self.stratify else None

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=stratify_col,
            random_state=self.random_state
        )

        print(f"   Test set: {len(test_df)} images ({self.test_size*100:.1f}%)")

        # Second split: create validation set from training data
        stratify_col_train = train_val_df['label'] if self.stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            stratify=stratify_col_train,
            random_state=self.random_state
        )

        print(f"   Training set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")

        # Verify no overlap
        assert len(set(train_df.index) & set(val_df.index)) == 0, "Train/val overlap!"
        assert len(set(train_df.index) & set(test_df.index)) == 0, "Train/test overlap!"
        assert len(set(val_df.index) & set(test_df.index)) == 0, "Val/test overlap!"

        print("   ✓ No data leakage detected")

        return train_df, val_df, test_df

    def _person_based_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Person-based split: different people in train/val/test.

        This is the GOLD STANDARD for evaluating generalization to new users.
        """
        unique_persons = df['person_id'].dropna().unique()
        n_persons = len(unique_persons)

        if n_persons < 10:
            print(f"   ⚠️  Warning: Only {n_persons} persons detected. Consider random split.")

        # Split persons into train/val/test
        train_persons, test_persons = train_test_split(
            unique_persons,
            test_size=self.test_size,
            random_state=self.random_state
        )

        train_persons, val_persons = train_test_split(
            train_persons,
            test_size=self.val_size,
            random_state=self.random_state
        )

        # Assign images based on person
        train_df = df[df['person_id'].isin(train_persons)]
        val_df = df[df['person_id'].isin(val_persons)]
        test_df = df[df['person_id'].isin(test_persons)]

        print(f"   Training set: {len(train_df)} images from {len(train_persons)} persons")
        print(f"   Validation set: {len(val_df)} images from {len(val_persons)} persons")
        print(f"   Test set: {len(test_df)} images from {len(test_persons)} persons")

        return train_df, val_df, test_df

    def load_images(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess images.

        Args:
            df: DataFrame with image paths and labels
            normalize: Whether to normalize pixel values to [0, 1]
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = df['label'].values

        iterator = tqdm(df['image_path'], desc="Loading images") if show_progress else df['image_path']

        for img_path in iterator:
            img = cv2.imread(img_path)

            if img is None:
                print(f"   ⚠️  Warning: Could not load {img_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size))

            # Normalize
            if normalize:
                img = img.astype(np.float32) / 255.0

            images.append(img)

        images = np.array(images, dtype=np.float32)

        # Fit label binarizer on first call
        if not hasattr(self.label_binarizer, 'classes_'):
            labels_encoded = self.label_binarizer.fit_transform(labels)
        else:
            labels_encoded = self.label_binarizer.transform(labels)

        return images, labels_encoded

    def load_all_splits(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load complete dataset with proper train/val/test splits.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Load dataset info
        df = self.load_dataset_info()

        # Create splits
        train_df, val_df, test_df = self.create_splits(df)

        # Load images
        print("\n📥 Loading training images...")
        X_train, y_train = self.load_images(train_df, normalize=normalize)

        print("📥 Loading validation images...")
        X_val, y_val = self.load_images(val_df, normalize=normalize)

        print("📥 Loading test images...")
        X_test, y_test = self.load_images(test_df, normalize=normalize)

        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Training set:   {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set:       {X_test.shape}")
        print(f"Image size:     {self.image_size}x{self.image_size}")
        print(f"Num classes:    {len(self.label_binarizer.classes_)}")
        print(f"Classes:        {list(self.label_binarizer.classes_)}")
        print("="*60 + "\n")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.label_binarizer.classes_)

    def get_class_distribution(self, split: str = "all") -> Dict[str, int]:
        """
        Get class distribution for analysis.

        Args:
            split: 'all', 'train', 'val', or 'test'

        Returns:
            Dictionary of class counts
        """
        return self.class_distribution
