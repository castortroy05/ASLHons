"""
Model architecture for ASL recognition.

This module fixes critical issues from the original implementation:
1. Proper input size (224x224 instead of 64x64)
2. Frozen base layers for transfer learning
3. Simpler, more appropriate head architecture
4. Support for modern architectures (EfficientNet, MobileNet)
5. Optional MediaPipe landmark integration
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
    MobileNetV3Small, MobileNetV3Large,
    ResNet50, ResNet101,
    VGG19
)
from typing import Tuple, List, Optional, Dict, Any


class ASLModelBuilder:
    """Builds ASL recognition models with proper transfer learning."""

    AVAILABLE_ARCHITECTURES = {
        'efficientnetv2b0': EfficientNetV2B0,
        'efficientnetv2b1': EfficientNetV2B1,
        'efficientnetv2b2': EfficientNetV2B2,
        'efficientnetv2': EfficientNetV2B0,  # Alias
        'mobilenetv3small': MobileNetV3Small,
        'mobilenetv3large': MobileNetV3Large,
        'mobilenetv3': MobileNetV3Large,  # Alias
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet': ResNet50,  # Alias
        'vgg19': VGG19,
    }

    def __init__(
        self,
        architecture: str = 'efficientnetv2',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 24,
        use_pretrained: bool = True,
        pretrained_weights: str = 'imagenet',
        freeze_base: bool = True,
        freeze_until_layer: int = -30,
    ):
        """
        Initialize model builder.

        Args:
            architecture: Model architecture name
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            use_pretrained: Whether to use pretrained weights
            pretrained_weights: 'imagenet' or path to weights
            freeze_base: Whether to freeze base model layers
            freeze_until_layer: Freeze layers up to this index (negative = from end)
        """
        self.architecture = architecture.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.pretrained_weights = pretrained_weights if use_pretrained else None
        self.freeze_base = freeze_base
        self.freeze_until_layer = freeze_until_layer

        # Validate architecture
        if self.architecture not in self.AVAILABLE_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list(self.AVAILABLE_ARCHITECTURES.keys())}"
            )

    def build(
        self,
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = 'relu',
    ) -> Model:
        """
        Build the model.

        Args:
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function

        Returns:
            Compiled Keras model
        """
        print(f"\n🏗️  Building {self.architecture.upper()} model...")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output classes: {self.num_classes}")
        print(f"   Pretrained: {self.use_pretrained}")
        print(f"   Freeze base: {self.freeze_base}")

        # Get base model
        base_model_class = self.AVAILABLE_ARCHITECTURES[self.architecture]

        base_model = base_model_class(
            include_top=False,
            weights=self.pretrained_weights,
            input_shape=self.input_shape,
            pooling='avg'  # Global average pooling
        )

        # Freeze base model if specified
        if self.freeze_base:
            if self.freeze_until_layer != 0:
                # Freeze up to a certain layer
                for layer in base_model.layers[:self.freeze_until_layer]:
                    layer.trainable = False
                print(f"   Frozen layers: 0 to {self.freeze_until_layer}")
            else:
                # Freeze all base layers
                base_model.trainable = False
                print(f"   All base layers frozen")
        else:
            print(f"   All base layers trainable")

        # Build custom head
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)

        # Add custom classification head
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation=activation, name=f'dense_{i}')(x)

            if use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_{i}')(x)

            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name=f'ASL_{self.architecture}')

        # Print model summary
        print(f"\n   Model summary:")
        trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        print(f"   Total parameters: {trainable_count + non_trainable_count:,}")
        print(f"   Trainable parameters: {trainable_count:,}")
        print(f"   Non-trainable parameters: {non_trainable_count:,}")

        return model

    def build_with_landmarks(
        self,
        landmark_input_size: int = 63,  # 21 keypoints × 3 coordinates
        dense_units: List[int] = [256, 128],
        dropout_rate: float = 0.5,
        combine_strategy: str = 'concat'
    ) -> Model:
        """
        Build model with MediaPipe landmarks + image features.

        Args:
            landmark_input_size: Size of landmark feature vector
            dense_units: Dense layer units
            dropout_rate: Dropout rate
            combine_strategy: How to combine features ('concat', 'add', 'multiply')

        Returns:
            Dual-input Keras model
        """
        print(f"\n🏗️  Building HYBRID model ({self.architecture} + MediaPipe)...")

        # Image branch
        base_model_class = self.AVAILABLE_ARCHITECTURES[self.architecture]
        base_model = base_model_class(
            include_top=False,
            weights=self.pretrained_weights,
            input_shape=self.input_shape,
            pooling='avg'
        )

        if self.freeze_base:
            base_model.trainable = False

        image_input = keras.Input(shape=self.input_shape, name='image_input')
        image_features = base_model(image_input, training=False)

        # Landmark branch
        landmark_input = keras.Input(shape=(landmark_input_size,), name='landmark_input')
        landmark_features = layers.Dense(128, activation='relu')(landmark_input)
        landmark_features = layers.BatchNormalization()(landmark_features)
        landmark_features = layers.Dropout(0.3)(landmark_features)

        # Combine features
        if combine_strategy == 'concat':
            combined = layers.concatenate([image_features, landmark_features])
        elif combine_strategy == 'add':
            # Need same dimension
            image_proj = layers.Dense(128)(image_features)
            combined = layers.add([image_proj, landmark_features])
        elif combine_strategy == 'multiply':
            image_proj = layers.Dense(128)(image_features)
            combined = layers.multiply([image_proj, landmark_features])
        else:
            raise ValueError(f"Unknown combine_strategy: {combine_strategy}")

        # Classification head
        x = combined
        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, landmark_input],
            outputs=outputs,
            name=f'ASL_{self.architecture}_hybrid'
        )

        print(f"✓ Hybrid model built with {combine_strategy} strategy")

        return model

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> 'ASLModelBuilder':
        """
        Create model builder from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ASLModelBuilder instance
        """
        model_config = config.get('model', {})
        data_config = config.get('data', {})

        image_size = data_config.get('image_size', 224)
        input_channels = data_config.get('input_channels', 3)

        return ASLModelBuilder(
            architecture=model_config.get('architecture', 'efficientnetv2'),
            input_shape=(image_size, image_size, input_channels),
            num_classes=data_config.get('num_classes', 24),
            use_pretrained=model_config.get('use_pretrained', True),
            pretrained_weights=model_config.get('pretrained_weights', 'imagenet'),
            freeze_base=model_config.get('freeze_base', True),
            freeze_until_layer=model_config.get('freeze_until_layer', -30),
        )


def compile_model(
    model: Model,
    optimizer: str = 'adam',
    learning_rate: float = 0.0001,
    loss: str = 'categorical_crossentropy',
    metrics: List[str] = None
) -> Model:
    """
    Compile the model.

    Args:
        model: Keras model to compile
        optimizer: Optimizer name
        learning_rate: Learning rate
        loss: Loss function
        metrics: List of metrics

    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]

    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    print(f"✓ Model compiled with {optimizer} (lr={learning_rate})")

    return model
