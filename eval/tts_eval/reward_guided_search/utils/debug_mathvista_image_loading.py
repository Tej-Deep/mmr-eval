#!/usr/bin/env python3
"""
Debug script to investigate MathVista image loading issues.
This script checks what's happening with image data when loading the mathvista_testmini dataset.
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
from mathvista_helper_functions import load_mathvista_dataset
import numpy as np

def debug_huggingface_dataset():
    """Debug the raw HuggingFace dataset to see what image fields are available."""
    print("=" * 60)
    print("DEBUGGING RAW HUGGINGFACE DATASET")
    print("=" * 60)
    
    # Load raw dataset from HuggingFace
    try:
        dataset = load_dataset('AI4Math/MathVista', split='testmini')
        print(f"✓ Successfully loaded {len(dataset)} items from HuggingFace dataset")
        # Use .select() to get first 3 rows, not columns
        data_subset = dataset.select(range(3))
        data_records = []
        for i, item in enumerate(data_subset):
            try:
                print(f"Processing item {i}, type: {type(item)}")
                if isinstance(item, str):
                    print(f"ERROR: Item {i} is a string: {item[:100]}...")
                    continue
                    
                record = {}
                
                # Basic fields with error handling
                try:
                    record['index'] = item['pid']
                    record['id'] = item['pid']
                except Exception as e:
                    print(f"ERROR accessing pid in item {i}: {e}")
                    continue
                    
                try:
                    record['question'] = item['question']
                    record['answer'] = item['answer']
                    record['question_type'] = item['question_type'] 
                    record['answer_type'] = item['answer_type']
                except Exception as e:
                    print(f"ERROR accessing basic fields in item {i}: {e}")
                    continue
                
                # Optional fields
                record['choices'] = item.get('choices', [])
                record['unit'] = item.get('unit', '')
                record['precision'] = item.get('precision', None)
                
                # Metadata fields with error handling
                try:
                    if 'metadata' in item and item['metadata']:
                        metadata = item['metadata']
                        record['grade'] = metadata.get('grade', '')
                        record['subject'] = metadata.get('subject', '')
                        record['category'] = metadata.get('category', '')
                        record['skills'] = metadata.get('skills', [])
                        record['source'] = metadata.get('source', '')
                        record['context'] = metadata.get('context', '')
                        record['task'] = metadata.get('task', '')
                        record['language'] = metadata.get('language', '')
                    else:
                        print(f"No metadata in item {i}")
                        record.update({
                            'grade': '', 'subject': '', 'category': '', 'skills': [],
                            'source': '', 'context': '', 'task': '', 'language': ''
                        })
                except Exception as e:
                    print(f"ERROR accessing metadata in item {i}: {e}")
                    record.update({
                        'grade': '', 'subject': '', 'category': '', 'skills': [],
                        'source': '', 'context': '', 'task': '', 'language': ''
                    })
                
                # Image processing with error handling
                try:
                    if 'decoded_image' in item and item['decoded_image'] is not None:
                        import io
                        import base64
                        buffered = io.BytesIO()
                        item['decoded_image'].save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        record['image'] = img_base64
                        print(f"✓ Converted image for item {i}")
                    else:
                        record['image'] = None
                        print(f"No decoded_image in item {i}")
                except Exception as e:
                    print(f"ERROR processing image in item {i}: {e}")
                    record['image'] = None
                    
                data_records.append(record)
                print(f"✓ Successfully processed item {i}")
                
            except Exception as e:
                print(f"ERROR processing item {i}: {e}")
                print(f"Item keys: {list(item.keys()) if hasattr(item, 'keys') else 'No keys method'}")
                continue
        
        # Convert to DataFrame
        data_df = pd.DataFrame(data_records)
        print(f"Loaded {len(data_df)} MathVista samples")
        return data_df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def test_load_mathvista_dataset():
    """Test the load_mathvista_dataset function exactly as used in vllm_bon_greedy_search_no_template.py"""
    print("=" * 60)
    print("TESTING load_mathvista_dataset FUNCTION - MIRRORING MAIN SCRIPT")
    print("=" * 60)

    try:
        # Mirror the exact loading from main script (lines 316-317)
        print("Loading MathVista dataset...")
        dataset_df = load_mathvista_dataset("AI4Math/MathVista", "testmini")
        print(
            f"✓ Successfully loaded {len(dataset_df)} items from load_mathvista_dataset()"
        )

        # Mirror the dataset assignment (line 334)
        dataset = dataset_df
        print(f"Dataset columns: {list(dataset.columns)}")
        print(f"Dataset shape: {dataset.shape}")

        # Mirror the data slicing for testing first few items (lines 387-388)
        start_idx = 0
        end_idx = min(3, len(dataset))  # Test first 3 items
        data = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
        print(
            f"Using range [{start_idx}, {end_idx}): Data size after selection: {len(data)}"
        )

        # Mirror the exact processing loop from main script (lines 442-467)
        for i in range(len(data)):
            print(f"\n--- Processing item {i} (mirroring main script) ---")
            line = data.iloc[i]

            # Mirror line_dict conversion (lines 447-452)
            line_dict = line.to_dict()
            for k, v in line_dict.items():
                if isinstance(v, np.integer):
                    line_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    line_dict[k] = float(v)

            print(f"  ID: {line_dict.get('id', 'N/A')}")
            print(f"  Question: {line_dict.get('question', 'N/A')[:100]}...")
            print(f"  Answer: {line_dict.get('answer', 'N/A')}")

            # Mirror the exact image extraction logic (lines 454-463)
            image_data_base64_list = []  # just a list of b64 strings, not objects
            if "image" in line_dict and line_dict["image"]:
                # Handle single image case
                if isinstance(line_dict["image"], str):
                    image_data_base64_list.append(line_dict["image"])
                # Handle multiple images case
                elif isinstance(line_dict["image"], list):
                    image_data_base64_list.extend(
                        [img for img in line_dict["image"] if img]
                    )

            print(f"  Found {len(image_data_base64_list)} base64 images in data")

            # Mirror the validation check (lines 466-467)
            if len(image_data_base64_list) == 0:
                print(f"  ⚠️  WARNING: No image data found for data sample {i}")
                print(f"  Image field type: {type(line_dict.get('image'))}")
                print(f"  Image field value: {line_dict.get('image')}")
            else:
                # Show image details like main script would use
                for j, img_b64 in enumerate(image_data_base64_list):
                    if img_b64:
                        print(
                            f"    Image {j}: length={len(img_b64)} chars, preview={img_b64[:50]}..."
                        )
                    else:
                        print(f"    Image {j}: None/empty")

        return dataset_df

    except Exception as e:
        print(f"✗ Error testing load_mathvista_dataset: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main debug function"""
    print("MathVista Image Loading Debug Script")
    print("====================================")

    # Test raw HuggingFace dataset
    # hf_df = debug_huggingface_dataset()
    # if hf_df is not None and len(hf_df) > 0:
    #     print(
    #         f"\nRaw HF dataset first image preview: {hf_df['image'][0][:50] if hf_df['image'][0] else 'None'}..."
    #     )

    # Test helper function
    helper_df = test_load_mathvista_dataset()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # if hf_df is not None and helper_df is not None:
    #     print("✓ Both tests completed successfully")
    #     print(f"Raw HF dataset: {len(hf_df)} items")
    #     print(f"Helper function: {len(helper_df)} items")
    # else:
    #     print("✗ Some tests failed - check errors above")

if __name__ == "__main__":
    main()