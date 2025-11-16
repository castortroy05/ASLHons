"""
Multi-Language Sign Language Manager

Handles loading and switching between different sign languages (ASL, BSL, ISL, etc.)
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging


class SignLanguageManager:
    """
    Manages multiple sign language configurations and models
    """

    def __init__(self, config_path: str = "configs/languages.yaml"):
        """
        Initialize language manager

        Args:
            config_path: Path to languages configuration file
        """
        self.config_path = Path(config_path)
        self.languages = self._load_languages()
        self.current_language = self.languages.get(
            self._get_default_language(), 'ASL'
        )
        self.loaded_models = {}

        logging.info(f"Initialized with {len(self.languages)} languages")

    def _load_languages(self) -> Dict:
        """
        Load language configurations from YAML

        Returns:
            Dictionary of language configurations
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Language config not found: {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('languages', {})

    def _get_default_language(self) -> str:
        """Get default language from config"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('default_language', 'ASL')

    def get_available_languages(self) -> List[str]:
        """
        Get list of available sign languages

        Returns:
            List of language codes (e.g., ['ASL', 'BSL', 'ISL'])
        """
        return list(self.languages.keys())

    def get_language_info(self, language_code: str) -> Dict:
        """
        Get information about a specific language

        Args:
            language_code: Language code (e.g., 'ASL')

        Returns:
            Dictionary with language information
        """
        if language_code not in self.languages:
            raise ValueError(
                f"Language {language_code} not found. "
                f"Available: {self.get_available_languages()}"
            )

        return self.languages[language_code]

    def switch_language(self, language_code: str) -> bool:
        """
        Switch to a different sign language

        Args:
            language_code: Target language code

        Returns:
            True if successful, False otherwise
        """
        if language_code not in self.languages:
            logging.error(
                f"Cannot switch to {language_code}. "
                f"Available: {self.get_available_languages()}"
            )
            return False

        self.current_language = language_code
        logging.info(
            f"Switched to {self.languages[language_code]['full_name']}"
        )
        return True

    def get_current_language(self) -> str:
        """Get current language code"""
        return self.current_language

    def get_vocabulary_size(self, language_code: Optional[str] = None) -> int:
        """
        Get vocabulary size for a language

        Args:
            language_code: Language code (uses current if None)

        Returns:
            Vocabulary size
        """
        lang = language_code or self.current_language
        return self.languages[lang].get('vocabulary_size', 0)

    def uses_fingerspelling(self, language_code: Optional[str] = None) -> bool:
        """
        Check if language uses fingerspelling

        Args:
            language_code: Language code (uses current if None)

        Returns:
            True if fingerspelling is used
        """
        lang = language_code or self.current_language
        return self.languages[lang].get('fingerspelling', False)

    def get_grammar_type(self, language_code: Optional[str] = None) -> str:
        """
        Get grammar type for a language

        Args:
            language_code: Language code (uses current if None)

        Returns:
            Grammar type (e.g., 'topic_comment', 'sov')
        """
        lang = language_code or self.current_language
        return self.languages[lang].get('grammar_type', 'unknown')

    def get_model_config(self, language_code: Optional[str] = None) -> Dict:
        """
        Get model configuration for a language

        Args:
            language_code: Language code (uses current if None)

        Returns:
            Model configuration dictionary
        """
        lang = language_code or self.current_language
        return self.languages[lang].get('model', {})

    def get_datasets(self, language_code: Optional[str] = None) -> List[Dict]:
        """
        Get available datasets for a language

        Args:
            language_code: Language code (uses current if None)

        Returns:
            List of dataset information
        """
        lang = language_code or self.current_language
        return self.languages[lang].get('datasets', [])

    def get_related_languages(
        self,
        language_code: Optional[str] = None
    ) -> List[str]:
        """
        Get related/similar sign languages

        Args:
            language_code: Language code (uses current if None)

        Returns:
            List of related language codes
        """
        lang = language_code or self.current_language
        lang_info = self.languages[lang]

        # Check direct relations
        related = lang_info.get('related_languages', [])

        # Check language families
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        families = config.get('language_families', {})
        for family_name, members in families.items():
            if lang in members:
                related.extend([m for m in members if m != lang])

        return list(set(related))

    def can_transfer_learn(
        self,
        source_lang: str,
        target_lang: str
    ) -> bool:
        """
        Check if transfer learning is viable between languages

        Args:
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            True if languages are similar enough for transfer learning
        """
        # Check if they're in the same language family
        related = self.get_related_languages(source_lang)

        return target_lang in related

    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get importance weights for different sign features

        Returns:
            Dictionary of feature weights
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('feature_weights', {})

    def get_realtime_config(self) -> Dict:
        """
        Get real-time processing configuration

        Returns:
            Real-time configuration dictionary
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('realtime', {})

    def get_translation_config(self) -> Dict:
        """
        Get translation settings

        Returns:
            Translation configuration dictionary
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config.get('translation', {})

    def print_language_comparison(self, languages: List[str]):
        """
        Print comparison table of multiple languages

        Args:
            languages: List of language codes to compare
        """
        print("\n" + "="*80)
        print("SIGN LANGUAGE COMPARISON")
        print("="*80)

        # Headers
        attributes = [
            'full_name', 'region', 'vocabulary_size',
            'fingerspelling', 'grammar_type'
        ]

        for attr in attributes:
            print(f"\n{attr.upper().replace('_', ' ')}:")
            for lang in languages:
                if lang in self.languages:
                    value = self.languages[lang].get(attr, 'N/A')
                    print(f"  {lang:10} {value}")

        print("\n" + "="*80 + "\n")


class GrammarTransformer:
    """
    Transforms text from spoken language grammar to sign language grammar

    Different sign languages have different grammars:
    - ASL/BSL: Topic-comment structure
    - JSL: SOV (Subject-Object-Verb)
    - Spatial grammar, classifiers, non-manual markers
    """

    def __init__(self, language_code: str = 'ASL'):
        self.language_code = language_code
        self.language_manager = SignLanguageManager()
        self.grammar_type = self.language_manager.get_grammar_type(language_code)

    def transform(self, text: str) -> str:
        """
        Transform English text to sign language grammar

        Args:
            text: Input text in English

        Returns:
            Text transformed to sign language grammar
        """
        if self.grammar_type == 'topic_comment':
            return self._topic_comment_transform(text)
        elif self.grammar_type == 'sov':
            return self._sov_transform(text)
        else:
            return text

    def _topic_comment_transform(self, text: str) -> str:
        """
        Transform to topic-comment structure (ASL, BSL, etc.)

        Rules:
        - Remove articles (a, an, the)
        - Remove "to be" verbs (am, is, are, was, were)
        - Time markers go first
        - Topic before comment
        - No passive voice
        """
        # Simple rule-based transformation
        # In production, use NLP library or trained model

        transformations = {
            # Remove articles
            ' a ': ' ',
            ' an ': ' ',
            ' the ': ' ',

            # Remove "to be" verbs
            ' am ': ' ',
            ' is ': ' ',
            ' are ': ' ',
            ' was ': ' ',
            ' were ': ' ',
            ' be ': ' ',
            ' being ': ' ',
            ' been ': ' ',

            # Simplify auxiliary verbs
            ' have ': ' ',
            ' has ': ' ',
            ' had ': ' ',
        }

        text = ' ' + text.lower() + ' '
        for old, new in transformations.items():
            text = text.replace(old, new)

        # Clean up
        text = ' '.join(text.split())
        text = text.upper()

        return text

    def _sov_transform(self, text: str) -> str:
        """
        Transform to SOV structure (JSL, etc.)

        Example: "I eat apple" → "I apple eat"
        """
        # Simplified SOV transformation
        # In production, use proper parsing

        # This would require NLP parsing to identify S, O, V
        # For now, return simplified version
        return text.upper()

    def add_nonmanual_markers(self, text: str) -> Dict[str, str]:
        """
        Add non-manual marker annotations

        Args:
            text: Sign gloss text

        Returns:
            Dictionary with text and marker annotations
        """
        # Detect question words
        question_markers = ['WHAT', 'WHERE', 'WHO', 'WHEN', 'WHY', 'HOW']
        is_question = any(word in text.split() for word in question_markers)

        markers = {
            'text': text,
            'facial': 'eyebrow_raise' if is_question else 'neutral',
            'head': 'tilt_forward' if is_question else 'neutral',
            'body': 'lean_forward' if is_question else 'neutral'
        }

        return markers


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = SignLanguageManager()

    # List available languages
    print(f"Available languages: {manager.get_available_languages()}")

    # Get info about ASL
    asl_info = manager.get_language_info('ASL')
    print(f"\nASL Info:")
    print(f"  Full name: {asl_info['full_name']}")
    print(f"  Vocabulary: {asl_info['vocabulary_size']} signs")
    print(f"  Fingerspelling: {asl_info['fingerspelling']}")

    # Switch to BSL
    manager.switch_language('BSL')
    print(f"\nCurrent language: {manager.get_current_language()}")

    # Check if transfer learning is viable
    can_transfer = manager.can_transfer_learn('ASL', 'BSL')
    print(f"\nCan transfer from ASL to BSL: {can_transfer}")

    # Compare languages
    manager.print_language_comparison(['ASL', 'BSL', 'ISL', 'Auslan'])

    # Grammar transformation
    transformer = GrammarTransformer('ASL')
    english = "What is your name?"
    asl_gloss = transformer.transform(english)
    print(f"\nEnglish: {english}")
    print(f"ASL gloss: {asl_gloss}")

    markers = transformer.add_nonmanual_markers(asl_gloss)
    print(f"With markers: {markers}")
