"""
Improved ASL Recognition Training Script

This script fixes all critical issues from the original implementation:
1. Proper train/val/test splits
2. Data augmentation
3. Frozen base layers with appropriate input size
4. Comprehensive evaluation
5. Optional MediaPipe integration
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import our modules
from utils.config_loader import load_config
from utils.seed import set_seeds, get_device, enable_mixed_precision
from data.dataset_loader import ASLDatasetLoader
from data.augmentation import ASLAugmentation, preview_augmentations
from models.model_builder import ASLModelBuilder, compile_model
from models.mediapipe_extractor import MediaPipeHandExtractor, augment_with_landmarks
from training.evaluator import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--use-mediapipe', action='store_true',
                        help='Use MediaPipe landmarks (overrides config)')
    parser.add_argument('--preview-augmentation', action='store_true',
                        help='Preview augmentation and exit')
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)

    # Override config with command line args
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.use_mediapipe:
        config['model']['use_mediapipe_landmarks'] = True

    print(f"   Experiment: {config['experiment_name']}")

    # Set random seeds
    set_seeds(config['random_seed'])

    # Configure device
    device = get_device(config['training'].get('device', 'auto'))
    enable_mixed_precision(config['training'].get('mixed_precision', False))

    # Load dataset
    print(f"\n{'='*60}")
    print("STEP 1: LOADING DATASET")
    print(f"{'='*60}")

    loader = ASLDatasetLoader(
        dataset_path=config['data']['dataset_path'],
        image_size=config['data']['image_size'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        stratify=config['data']['stratify'],
        random_state=config['random_seed'],
        person_based_split=config['data'].get('person_based_split', False)
    )

    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_all_splits(
        normalize=config['data']['normalize']
    )

    class_names = loader.get_class_names()

    # Setup data augmentation
    print(f"\n{'='*60}")
    print("STEP 2: SETTING UP DATA AUGMENTATION")
    print(f"{'='*60}")

    augmentation = ASLAugmentation.create_from_config(config)

    # Preview augmentation if requested
    if args.preview_augmentation:
        print("\n📸 Previewing augmentation effects...")
        preview_augmentations(X_train[:5], y_train[:5], augmentation)
        print("✓ Preview complete. Exiting.")
        return

    # Extract MediaPipe landmarks if enabled
    use_landmarks = config['model'].get('use_mediapipe_landmarks', False)

    if use_landmarks:
        print(f"\n{'='*60}")
        print("STEP 3: EXTRACTING MEDIAPIPE LANDMARKS")
        print(f"{'='*60}")

        with MediaPipeHandExtractor() as extractor:
            print("\nExtracting training set landmarks...")
            _, landmarks_train = augment_with_landmarks(X_train, extractor)

            print("Extracting validation set landmarks...")
            _, landmarks_val = augment_with_landmarks(X_val, extractor)

            print("Extracting test set landmarks...")
            _, landmarks_test = augment_with_landmarks(X_test, extractor)

        print(f"✓ Landmark extraction complete")
        print(f"   Training landmarks: {landmarks_train.shape}")
        print(f"   Validation landmarks: {landmarks_val.shape}")
        print(f"   Test landmarks: {landmarks_test.shape}")

    # Build model
    print(f"\n{'='*60}")
    print(f"STEP {'4' if use_landmarks else '3'}: BUILDING MODEL")
    print(f"{'='*60}")

    model_builder = ASLModelBuilder.create_from_config(config)

    if use_landmarks:
        # Build hybrid model (image + landmarks)
        model = model_builder.build_with_landmarks(
            landmark_input_size=landmarks_train.shape[1],
            dense_units=config['model']['dense_units'],
            dropout_rate=config['model']['dropout_rate'],
            combine_strategy=config['model'].get('combine_strategy', 'concat')
        )
    else:
        # Build image-only model
        model = model_builder.build(
            dense_units=config['model']['dense_units'],
            dropout_rate=config['model']['dropout_rate'],
            use_batch_norm=config['model']['use_batch_norm'],
            activation=config['model']['activation']
        )

    # Compile model
    model = compile_model(
        model,
        optimizer=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )

    # Setup callbacks
    callbacks = []

    # Early stopping
    if config['training']['callbacks']['early_stopping']['enabled']:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=config['training']['callbacks']['early_stopping']['monitor'],
            patience=config['training']['callbacks']['early_stopping']['patience'],
            restore_best_weights=config['training']['callbacks']['early_stopping']['restore_best_weights'],
            min_delta=config['training']['callbacks']['early_stopping']['min_delta'],
            verbose=1
        ))

    # Reduce learning rate
    if config['training']['callbacks']['reduce_lr']['enabled']:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=config['training']['callbacks']['reduce_lr']['monitor'],
            factor=config['training']['callbacks']['reduce_lr']['factor'],
            patience=config['training']['callbacks']['reduce_lr']['patience'],
            min_lr=config['training']['callbacks']['reduce_lr']['min_lr'],
            verbose=1
        ))

    # Model checkpoint
    if config['training']['callbacks']['model_checkpoint']['enabled']:
        checkpoint_path = Path(config['paths']['checkpoint_dir']) / f"{config['experiment_name']}_best.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config['training']['callbacks']['model_checkpoint']['monitor'],
            save_best_only=config['training']['callbacks']['model_checkpoint']['save_best_only'],
            save_weights_only=config['training']['callbacks']['model_checkpoint']['save_weights_only'],
            mode=config['training']['callbacks']['model_checkpoint']['mode'],
            verbose=1
        ))

    # TensorBoard
    if config['training']['callbacks']['tensorboard']['enabled']:
        log_dir = Path(config['paths']['tensorboard_dir']) / config['experiment_name']
        log_dir.mkdir(parents=True, exist_ok=True)

        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=config['training']['callbacks']['tensorboard']['histogram_freq'],
            write_graph=config['training']['callbacks']['tensorboard']['write_graph'],
            write_images=config['training']['callbacks']['tensorboard']['write_images'],
            update_freq=config['training']['callbacks']['tensorboard']['update_freq']
        ))

    # Training
    print(f"\n{'='*60}")
    print(f"STEP {'5' if use_landmarks else '4'}: TRAINING MODEL")
    print(f"{'='*60}")

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']

    print(f"\n   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Callbacks: {len(callbacks)} enabled")
    print()

    with tf.device(device):
        if use_landmarks:
            # Train with dual inputs (images + landmarks)
            history = model.fit(
                [X_train, landmarks_train],
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([X_val, landmarks_val], y_val),
                callbacks=callbacks,
                workers=config['training']['workers'],
                use_multiprocessing=config['training']['use_multiprocessing'],
                verbose=1
            )
        else:
            # Train with images only (using augmentation)
            train_gen = augmentation.flow(X_train, y_train, batch_size=batch_size, augment=True)

            history = model.fit(
                train_gen,
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                workers=config['training']['workers'],
                use_multiprocessing=config['training']['use_multiprocessing'],
                verbose=1
            )

    # Evaluation
    print(f"\n{'='*60}")
    print(f"STEP {'6' if use_landmarks else '5'}: EVALUATION")
    print(f"{'='*60}")

    # Create evaluator
    evaluator = ModelEvaluator(
        class_names=class_names,
        output_dir=Path(config['paths']['metrics_dir']) / config['experiment_name']
    )

    # Predict on test set
    if use_landmarks:
        y_pred_proba = model.predict([X_test, landmarks_test])
    else:
        y_pred_proba = model.predict(X_test)

    y_pred = (y_pred_proba == y_pred_proba.max(axis=1, keepdims=True)).astype(int)

    # Generate comprehensive report
    evaluator.generate_report(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        X=X_test,
        history=history.history
    )

    # Save final model
    final_model_path = Path(config['paths']['checkpoint_dir']) / f"{config['experiment_name']}_final.h5"
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n   Experiment: {config['experiment_name']}")
    print(f"   Final Test Accuracy: {evaluator.metrics['accuracy']:.4f}")
    print(f"   Outputs saved to: {config['paths']['metrics_dir']}/{config['experiment_name']}/")
    print()


if __name__ == '__main__':
    main()
