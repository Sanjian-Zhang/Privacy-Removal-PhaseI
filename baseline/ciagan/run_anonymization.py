#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def run_anonymization(data_path, model_path, output_path, num_identities=1):
    print("=== CIAGAN Identity Anonymization ===")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Number of identities: {num_identities}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file does not exist: {model_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ Data directory does not exist: {data_path}")
        return False
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        import sys
        import os
        source_path = os.path.join(os.path.dirname(__file__), 'source')
        sys.path.insert(0, source_path)
        
        from test import run_inference
        
        model_name = model_path.replace('.pth', '')
        
        run_inference(
            data_path=data_path,
            num_folders=num_identities,
            model_path=model_name,
            output_path=output_path
        )
        
        print("✅ Identity anonymization completed!")
        print(f"Results saved in: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIAGAN Identity Anonymization')
    parser.add_argument('--data', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/processed_output/',
                       help='Processed data path')
    parser.add_argument('--model', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/pretrained_models/modelG.pth',
                       help='Pretrained model path')
    parser.add_argument('--output', type=str,
                       default='/home/zhiqics/sanjian/baseline/ciagan/anonymized_output/',
                       help='Output path')
    parser.add_argument('--ids', type=int, default=1,
                       help='Number of identities')
    
    args = parser.parse_args()
    
    run_anonymization(args.data, args.model, args.output, args.ids)
