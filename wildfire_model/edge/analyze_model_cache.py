#!/usr/bin/env python3
"""
Check what models are in the cache and their actual properties
"""

import os
import sys
import tensorflow as tf
from datetime import datetime

def safe_count_params(model_or_weights):
    """Safely count parameters handling different TensorFlow versions"""
    try:
        if hasattr(model_or_weights, 'count_params'):
            return model_or_weights.count_params()
        elif isinstance(model_or_weights, list):
            total = 0
            for w in model_or_weights:
                if hasattr(w, 'shape'):
                    import numpy as np
                    total += np.prod(w.shape)
            return int(total)
        else:
            return 'Unknown'
    except Exception:
        return 'Unknown'

def analyze_model_file(filepath):
    """Analyze a model file and return its properties"""
    print(f"\n🔍 Analyzing: {filepath}")
    
    # Check file exists and size
    if not os.path.exists(filepath):
        print("   ❌ File not found")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"   📁 File size: {file_size / (1024*1024):.1f} MB")
    print(f"   📅 Modified: {datetime.fromtimestamp(os.path.getmtime(filepath))}")
    
    # Try to load and analyze
    try:
        print("   🔄 Loading model...")
        model = tf.keras.models.load_model(filepath)
        
        print(f"   ✅ Model loaded successfully")
        print(f"   🏗️  Model type: {type(model).__name__}")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
        print(f"   🔢 Total parameters: {safe_count_params(model)}")
        print(f"   🔢 Trainable parameters: {safe_count_params(model.trainable_weights)}")
        print(f"   📚 Number of layers: {len(model.layers)}")
        
        # Check if it's the dummy model pattern
        params = safe_count_params(model)
        if params == 2123813:
            print("   ⚠️  WARNING: This has DUMMY MODEL parameters!")
        elif str(model.input_shape) == "(None, 128, 128, 3)":
            print("   ⚠️  WARNING: This has DUMMY MODEL input shape!")
        else:
            print("   ✅ This appears to be a REAL model (different from dummy)")
            
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")

def main():
    print("🔍 Model Cache Analysis")
    print("=" * 50)
    
    # Check possible model cache locations
    cache_dirs = [
        "./models",
        "./edge/models", 
        "/tmp/models",
        "."
    ]
    
    model_files_found = []
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"\n📁 Checking directory: {cache_dir}")
            
            # Look for .h5 files
            for filename in os.listdir(cache_dir):
                if filename.endswith('.h5'):
                    filepath = os.path.join(cache_dir, filename)
                    model_files_found.append(filepath)
                    print(f"   📄 Found: {filename}")
    
    if not model_files_found:
        print("\n❌ No .h5 model files found in cache directories")
        print("   Possible locations to check manually:")
        for cache_dir in cache_dirs:
            print(f"   - {os.path.abspath(cache_dir)}")
    else:
        print(f"\n📊 Analyzing {len(model_files_found)} model files:")
        for filepath in model_files_found:
            analyze_model_file(filepath)
    
    # Check if we can find the specific models mentioned
    print("\n🎯 Looking for specific uploaded models:")
    specific_models = ["al_best.h5", "best_model.h5"]
    
    for model_name in specific_models:
        for cache_dir in cache_dirs:
            filepath = os.path.join(cache_dir, model_name)
            if os.path.exists(filepath):
                print(f"\n✅ Found {model_name} in {cache_dir}")
                analyze_model_file(filepath)
                break
        else:
            print(f"\n❌ {model_name} not found in any cache directory")
    
    print("\n" + "=" * 50)
    print("💡 If the real models exist but still show dummy characteristics,")
    print("   the issue is with the global model variable update in app.py")

if __name__ == "__main__":
    main()
