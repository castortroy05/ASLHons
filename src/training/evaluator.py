"""
Comprehensive evaluation and visualization for ASL recognition models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    top_k_accuracy_score
)
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
import json


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""

    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "outputs/metrics"
    ):
        """
        Initialize evaluator.

        Args:
            class_names: List of class names
            output_dir: Directory to save outputs
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Comprehensive evaluation.

        Args:
            y_true: True labels (one-hot encoded or indices)
            y_pred: Predicted labels (one-hot encoded or indices)
            y_pred_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Convert one-hot to indices if needed
        if y_true.ndim > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        if y_pred.ndim > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred

        # Compute metrics
        accuracy = accuracy_score(y_true_idx, y_pred_idx)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, average=None, labels=range(len(self.class_names))
        )

        # Overall metrics
        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()

        # Top-k accuracy (if probabilities available)
        if y_pred_proba is not None:
            top3_acc = top_k_accuracy_score(y_true_idx, y_pred_proba, k=3)
            top5_acc = top_k_accuracy_score(y_true_idx, y_pred_proba, k=5)
        else:
            top3_acc = None
            top5_acc = None

        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
        }

        # Print summary
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:      {accuracy:.4f}")
        print(f"  Precision:     {precision_macro:.4f}")
        print(f"  Recall:        {recall_macro:.4f}")
        print(f"  F1-Score:      {f1_macro:.4f}")

        if top3_acc is not None:
            print(f"  Top-3 Accuracy: {top3_acc:.4f}")
        if top5_acc is not None:
            print(f"  Top-5 Accuracy: {top5_acc:.4f}")

        print("\nPer-Class Metrics:")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)

        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")

        # Generate confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        self.metrics['confusion_matrix'] = cm.tolist()

        return self.metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize values
            save_path: Path to save figure
        """
        # Convert one-hot to indices if needed
        if y_true.ndim > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        if y_pred.ndim > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred

        # Compute confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )

        plt.title(title, fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'confusion_matrix.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()

    def plot_per_class_metrics(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot per-class precision, recall, F1-score.

        Args:
            save_path: Path to save figure
        """
        if 'per_class_precision' not in self.metrics:
            print("⚠️  No metrics available. Run evaluate() first.")
            return

        # Create DataFrame for easy plotting
        df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': self.metrics['per_class_precision'],
            'Recall': self.metrics['per_class_recall'],
            'F1-Score': self.metrics['per_class_f1'],
        })

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(self.class_names))
        width = 0.25

        ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'per_class_metrics.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics saved to {save_path}")
        plt.close()

    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot training history (loss and accuracy curves).

        Args:
            history: Training history from model.fit()
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)

        # Plot loss
        ax2.plot(history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'training_history.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
        plt.close()

    def visualize_errors(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Visualize misclassified examples.

        Args:
            X: Input images
            y_true: True labels
            y_pred: Predicted labels
            n_samples: Number of error samples to show
            save_path: Path to save figure
        """
        # Convert one-hot to indices if needed
        if y_true.ndim > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true

        if y_pred.ndim > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred

        # Find misclassified samples
        errors = np.where(y_true_idx != y_pred_idx)[0]

        if len(errors) == 0:
            print("✓ No errors to visualize - perfect predictions!")
            return

        n_samples = min(n_samples, len(errors))
        error_indices = np.random.choice(errors, n_samples, replace=False)

        # Plot
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for i, idx in enumerate(error_indices):
            ax = axes[i]

            img = X[idx]
            true_label = self.class_names[y_true_idx[idx]]
            pred_label = self.class_names[y_pred_idx[idx]]

            ax.imshow(img)
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10, color='red')
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / 'error_examples.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error examples saved to {save_path}")
        plt.close()

    def save_metrics(self, save_path: Optional[str] = None):
        """
        Save metrics to JSON file.

        Args:
            save_path: Path to save JSON
        """
        if save_path is None:
            save_path = self.output_dir / 'metrics.json'

        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"✓ Metrics saved to {save_path}")

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        history: Optional[Dict] = None
    ):
        """
        Generate complete evaluation report with all visualizations.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            X: Input images (for error visualization)
            history: Training history
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60 + "\n")

        # Compute metrics
        self.evaluate(y_true, y_pred, y_pred_proba)

        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(y_true, y_pred, normalize=True,
                                   save_path=self.output_dir / 'confusion_matrix_normalized.png')
        self.plot_per_class_metrics()

        if history is not None:
            self.plot_training_history(history)

        if X is not None:
            self.visualize_errors(X, y_true, y_pred)

        # Save metrics
        self.save_metrics()

        print("\n" + "="*60)
        print(f"✓ Evaluation report saved to {self.output_dir}/")
        print("="*60 + "\n")
