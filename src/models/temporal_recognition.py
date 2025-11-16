"""
Temporal Sign Language Recognition Models

Handles sequential sign language recognition using LSTM or Transformer architectures.
Goes beyond static fingerspelling to recognize full dynamic signs.
"""

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class TransformerSignRecognizer(nn.Module):
    """
    Transformer-based sign language recognition model

    Architecture:
    - Input: Sequence of MediaPipe landmarks (T, 543)
    - Positional encoding
    - Multi-head self-attention layers
    - Feed-forward layers
    - Output: Sign predictions
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_length: int = 300
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Input projection (landmarks to embedding)
        self.input_projection = nn.Linear(543, embedding_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, landmarks_sequence, mask=None):
        """
        Forward pass

        Args:
            landmarks_sequence: (batch, time, 543) - MediaPipe landmarks
            mask: Optional attention mask

        Returns:
            Sign probabilities: (batch, time, vocab_size)
        """
        # Project landmarks to embedding space
        x = self.input_projection(landmarks_sequence)  # (B, T, E)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Project to vocabulary
        logits = self.output_projection(x)  # (B, T, V)

        return logits


class LSTMSignRecognizer(keras.Model):
    """
    LSTM-based sign language recognition model (TensorFlow)

    Architecture:
    - Input: Sequence of MediaPipe landmarks
    - Bidirectional LSTM layers
    - Attention mechanism
    - Output layer with CTC loss
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        embedding_dim: int = 256,
        lstm_units: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # Input processing
        self.input_dense = layers.Dense(embedding_dim, activation='relu')
        self.input_dropout = layers.Dropout(dropout)

        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(
                        lstm_units,
                        return_sequences=True,
                        dropout=dropout
                    )
                )
            )

        # Attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=lstm_units * 2  # Bidirectional
        )

        # Output layer
        self.output_dense = layers.Dense(vocab_size + 1, activation='softmax')  # +1 for CTC blank

    def call(self, landmarks_sequence, training=False):
        """
        Forward pass

        Args:
            landmarks_sequence: (batch, time, 543)
            training: Training mode flag

        Returns:
            Sign probabilities: (batch, time, vocab_size + 1)
        """
        # Input projection
        x = self.input_dense(landmarks_sequence)
        x = self.input_dropout(x, training=training)

        # LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # Self-attention
        attention_output = self.attention(x, x, training=training)
        x = layers.Add()([x, attention_output])  # Residual connection

        # Output projection
        logits = self.output_dense(x)

        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class SignLanguageDecoder:
    """
    Decode model output to sign sequences

    Supports:
    - CTC decoding for continuous sequences
    - Beam search for better accuracy
    - Language model integration
    """

    def __init__(
        self,
        vocabulary: List[str],
        blank_idx: int = 0,
        beam_width: int = 10
    ):
        self.vocabulary = vocabulary
        self.blank_idx = blank_idx
        self.beam_width = beam_width

        # Create character to index mapping
        self.char2idx = {char: idx for idx, char in enumerate(vocabulary)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def ctc_decode(self, logits: np.ndarray) -> List[str]:
        """
        CTC greedy decoding

        Args:
            logits: (time, vocab_size) - Model output

        Returns:
            Decoded sign sequence
        """
        # Get most likely class at each timestep
        predictions = np.argmax(logits, axis=-1)

        # Collapse repeats and remove blanks
        decoded = []
        prev_char = None

        for pred in predictions:
            if pred != self.blank_idx and pred != prev_char:
                if pred < len(self.idx2char):
                    decoded.append(self.idx2char[pred])
            prev_char = pred

        return decoded

    def beam_search_decode(
        self,
        logits: np.ndarray,
        beam_width: int = None
    ) -> List[Tuple[List[str], float]]:
        """
        CTC beam search decoding

        Args:
            logits: (time, vocab_size) - Model output
            beam_width: Number of beams to keep

        Returns:
            List of (decoded_sequence, score) tuples
        """
        if beam_width is None:
            beam_width = self.beam_width

        # Initialize beam with empty sequence
        beams = [([''], 0.0)]  # (sequence, score)

        for t in range(logits.shape[0]):
            new_beams = []

            for sequence, score in beams:
                # Get top-k predictions at this timestep
                top_k_probs = np.log(np.sort(logits[t])[::-1][:beam_width])
                top_k_indices = np.argsort(logits[t])[::-1][:beam_width]

                for prob, idx in zip(top_k_probs, top_k_indices):
                    if idx == self.blank_idx:
                        # Blank - keep sequence as is
                        new_beams.append((sequence, score + prob))
                    elif idx < len(self.idx2char):
                        # Add new character
                        new_sequence = sequence + [self.idx2char[idx]]
                        new_beams.append((new_sequence, score + prob))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        return beams


class RealtimeSignRecognizer:
    """
    Real-time sign language recognition from video stream

    Features:
    - Sliding window processing
    - Temporal smoothing
    - Confidence thresholding
    - Multi-language support
    """

    def __init__(
        self,
        model: nn.Module,
        landmark_extractor,
        vocabulary: List[str],
        language: str = 'ASL',
        buffer_size: int = 90,  # 3 seconds at 30 FPS
        confidence_threshold: float = 0.7
    ):
        self.model = model
        self.landmark_extractor = landmark_extractor
        self.decoder = SignLanguageDecoder(vocabulary)
        self.language = language
        self.confidence_threshold = confidence_threshold

        # Frame buffer for temporal context
        self.frame_buffer = deque(maxlen=buffer_size)
        self.landmark_buffer = deque(maxlen=buffer_size)

        # Smoothing for predictions
        self.prediction_history = deque(maxlen=5)

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process single video frame

        Args:
            frame: RGB image

        Returns:
            Recognition result with confidence
        """
        # Extract landmarks
        landmarks = self.landmark_extractor.extract(frame)

        if landmarks is None:
            return None

        # Add to buffer
        self.frame_buffer.append(frame)
        self.landmark_buffer.append(landmarks)

        # Process when buffer is full
        if len(self.landmark_buffer) == self.landmark_buffer.maxlen:
            return self._recognize_sequence()

        return None

    def _recognize_sequence(self) -> Dict:
        """
        Recognize sign from buffered sequence

        Returns:
            Dictionary with sign, confidence, timestamp
        """
        # Stack landmarks into sequence
        landmarks_sequence = np.stack(list(self.landmark_buffer))
        landmarks_tensor = torch.FloatTensor(landmarks_sequence).unsqueeze(0)

        # Run model
        with torch.no_grad():
            logits = self.model(landmarks_tensor)

        # Decode
        predictions = self.decoder.ctc_decode(logits[0].cpu().numpy())

        # Get confidence
        probabilities = torch.softmax(logits, dim=-1)
        confidence = torch.max(probabilities).item()

        # Smooth predictions
        self.prediction_history.append((predictions, confidence))
        smoothed_prediction = self._smooth_predictions()

        return {
            'signs': smoothed_prediction,
            'confidence': confidence,
            'timestamp': len(self.prediction_history),
            'language': self.language
        }

    def _smooth_predictions(self) -> List[str]:
        """
        Smooth predictions using voting

        Returns:
            Most common prediction in recent history
        """
        if not self.prediction_history:
            return []

        # Weight recent predictions more heavily
        weighted_predictions = {}

        for i, (pred, conf) in enumerate(self.prediction_history):
            weight = (i + 1) / len(self.prediction_history) * conf
            pred_str = ' '.join(pred)

            if pred_str in weighted_predictions:
                weighted_predictions[pred_str] += weight
            else:
                weighted_predictions[pred_str] = weight

        # Return most weighted prediction
        if weighted_predictions:
            best_pred = max(weighted_predictions, key=weighted_predictions.get)
            return best_pred.split()

        return []


# Example usage
if __name__ == "__main__":
    # Example: Create transformer model
    model = TransformerSignRecognizer(
        vocab_size=2000,  # WLASL vocabulary
        embedding_dim=512,
        num_heads=8,
        num_layers=6
    )

    # Example input
    batch_size = 4
    seq_length = 100
    landmarks = torch.randn(batch_size, seq_length, 543)

    # Forward pass
    output = model(landmarks)
    print(f"Output shape: {output.shape}")  # (4, 100, 2000)

    # Example: LSTM model in TensorFlow
    lstm_model = LSTMSignRecognizer(
        vocab_size=2000,
        embedding_dim=256,
        lstm_units=512,
        num_layers=3
    )

    # Build model
    lstm_model.build(input_shape=(None, None, 543))
    lstm_model.summary()
