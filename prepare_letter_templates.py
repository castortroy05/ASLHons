"""
Utility script to prepare letter templates from the ASL dataset.

This creates representative images for each letter that can be used
for text-to-sign translation and avatar generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from text_to_sign.translator import create_letter_templates_from_dataset

if __name__ == '__main__':
    # Default paths
    dataset_path = "../ASLTransalation/fingerspelling/data"
    output_dir = "data/processed/letter_templates"

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please update the dataset_path in this script or provide it as an argument.")
        sys.exit(1)

    print("="*60)
    print("CREATING LETTER TEMPLATES")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_dir}")
    print()

    # Create templates
    create_letter_templates_from_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        samples_per_letter=5
    )

    print()
    print("="*60)
    print("✓ LETTER TEMPLATES CREATED")
    print("="*60)
    print(f"\nTemplates saved to: {output_dir}/")
    print("\nYou can now use these templates for:")
    print("  1. Text-to-sign translation")
    print("  2. Avatar generation")
    print("  3. Demo application")
    print()
