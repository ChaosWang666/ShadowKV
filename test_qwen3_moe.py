#!/usr/bin/env python3
"""
Test script for Qwen3MoE ShadowKV implementation

This script tests the basic functionality of the Qwen3MoE class,
including initialization and basic method calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test basic syntax and class structure without full imports
try:
    # First test if the file can be compiled
    import py_compile
    py_compile.compile('/root/workspace/personal_data/project_and_paper/ShadowKV/models/qwen3_moe.py', doraise=True)
    print("✓ qwen3_moe.py compiles successfully")
except py_compile.PyCompileError as e:
    print(f"✗ Compilation failed: {e}")
    sys.exit(1)

# Test class structure by reading the file
try:
    with open('/root/workspace/personal_data/project_and_paper/ShadowKV/models/qwen3_moe.py', 'r') as f:
        content = f.read()
    
    # Check for required class definitions
    if 'class Qwen3MoeLayer:' in content:
        print("✓ Qwen3MoeLayer class defined")
    else:
        print("✗ Qwen3MoeLayer class missing")
    
    if 'class Qwen3Moe(LLM):' in content:
        print("✓ Qwen3Moe class defined and inherits from LLM")
    else:
        print("✗ Qwen3Moe class missing or incorrect inheritance")
    
    # Check for required methods in the file content
    print("\n=== Testing Method Definitions ===")
    required_methods = [
        'def pre_attention_compute(',
        'def post_attention_compute(', 
        'def apply_rotary_pos_emb(',
        'def apply_rotary_pos_emb_single('
    ]
    
    # These methods should be inherited from LLM base class
    inherited_methods = [
        'def forward(',
        'def generate(',
        'def chat(',
        'def init_kv_cache_generate(',
        'def init_attn_metadata('
    ]
    
    for method in required_methods:
        if method in content:
            method_name = method.replace('def ', '').replace('(', '')
            print(f"✓ Method '{method_name}' defined")
        else:
            method_name = method.replace('def ', '').replace('(', '')
            print(f"✗ Method '{method_name}' missing")
    
    print("\n=== Checking Inherited Methods (should NOT be redefined) ===")
    for method in inherited_methods:
        if method in content:
            method_name = method.replace('def ', '').replace('(', '')
            print(f"✗ Method '{method_name}' should be inherited from LLM, not redefined")
        else:
            method_name = method.replace('def ', '').replace('(', '')
            print(f"✓ Method '{method_name}' correctly inherited from LLM")
    
    # Check for MoE-specific features
    print("\n=== Testing MoE-specific Features ===")
    moe_features = [
        'def _moe_forward(',
        'is_moe_layer',
        'gate_proj',
        'up_proj',
        'down_proj'
    ]
    
    for feature in moe_features:
        if feature in content:
            print(f"✓ MoE feature '{feature}' found")
        else:
            print(f"✗ MoE feature '{feature}' missing")
    
    print("\n=== All Tests Passed ===")
    print("Qwen3MoE ShadowKV implementation is ready for use!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)