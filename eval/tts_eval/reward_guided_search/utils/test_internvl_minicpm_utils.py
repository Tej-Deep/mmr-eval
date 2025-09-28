#!/usr/bin/env python3
"""Test the prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm function"""

import sys
sys.path.append('/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation')
from reward_guided_search.utils import prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm


def test_prepare_question_array():
    """Test the prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm function"""
    print("\n" + "="*100)
    print("TESTING prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm")
    print("="*100)
    
    # Create dummy base64 strings for testing
    img1_base64 = "dummy_base64_image_1"
    img2_base64 = "dummy_base64_image_2"
    img3_base64 = "dummy_base64_image_3"
    
    test_cases = [
        # Test 1: Single image, non-interleaved
        {
            "name": "Single image, non-interleaved",
            "question": "What is in this image?",
            "images": [img1_base64],
            "interleave": False
        },
        # Test 2: Multiple images, non-interleaved
        {
            "name": "Multiple images, non-interleaved",
            "question": "Compare these images and describe the differences.",
            "images": [img1_base64, img2_base64, img3_base64],
            "interleave": False
        },
        # Test 3: Single image, interleaved
        {
            "name": "Single image, interleaved",
            "question": "Look at <image 1> and describe what you see.",
            "images": [img1_base64],
            "interleave": True
        },
        # Test 4: Multiple images, interleaved
        {
            "name": "Multiple images, interleaved",
            "question": "Compare <image 1> with <image 2> and explain how <image 3> differs from both.",
            "images": [img1_base64, img2_base64, img3_base64],
            "interleave": True
        },
        # Test 5: Multiple occurrences of same image
        {
            "name": "Multiple occurrences of same image",
            "question": "First look at <image 1>, then <image 2>, now compare with <image 1> again. Also check <image 3>.",
            "images": [img1_base64, img2_base64, img3_base64],
            "interleave": True
        },
        # Test 6: Non-sequential image references
        {
            "name": "Non-sequential image references",
            "question": "Analyze <image 3> first, then <image 1>, and finally <image 2>.",
            "images": [img1_base64, img2_base64, img3_base64],
            "interleave": True
        },
    ]
    
    for test_case in test_cases:
        print(f"\n{'-'*80}")
        print(f"TEST: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print(f"Number of images: {len(test_case['images'])}")
        print(f"Interleaved: {test_case['interleave']}")
        print(f"{'-'*80}")
        
        for model_type in ["minicpm", "internvl"]:
            print(f"\nModel Type: {model_type.upper()}")
            
            result, output_images = prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm(
                question=test_case['question'],
                image_data_base64_list=test_case['images'],
                interleave_image_tokens=test_case['interleave'],
                model_type=model_type
            )
            
            print(f"Result structure:")
            for msg in result:
                print(f"  Role: {msg['role']}")
                print(f"  Content: {msg['content'][:200]}..." if len(str(msg['content'])) > 200 else f"  Content: {msg['content']}")
            
            print(f"Output images count: {len(output_images)}")
            if test_case['interleave'] and output_images:
                print(f"Output images: {output_images}")
    
    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    # Run the test function
    test_prepare_question_array()