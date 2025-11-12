#!/usr/bin/env python3
"""
Simple test script to verify the openCLT setup is working correctly.
This script performs a minimal test without requiring large datasets.
"""

import torch
from openclt import OpenCrossLayerTranscoder

def test_setup():
    print("Testing openCLT setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Test initialization
    print("Initializing OpenCrossLayerTranscoder...")
    try:
        transcoder = OpenCrossLayerTranscoder(
            model_name="gpt2",
            num_features=100,
            device=device,
            activation_type="topk",
            topk_features=5
        )
        print("✓ Transcoder initialized successfully!\n")
    except Exception as e:
        print(f"✗ Failed to initialize transcoder: {e}")
        return False
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        test_text = "Hello, world!"
        input_ids = transcoder.tokenizer.encode(test_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = transcoder(input_ids)
        
        print("✓ Forward pass successful!")
        print(f"  - Feature activations: {len(outputs['feature_activations'])} layers")
        print(f"  - Reconstructed activations: {len(outputs['reconstructed_activations'])} layers\n")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ All tests passed! The setup is working correctly.")
    print("\nYou can now run the full example with:")
    print("  poetry run python examples/practice_run.py --percent_samples 1")
    return True

if __name__ == "__main__":
    success = test_setup()
    exit(0 if success else 1)

