#!/usr/bin/env python3
"""
Extract all wrong-labeled queries from human-eval and organize by model
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def extract_wrong_samples(human_eval_folder: str = "human-eval") -> Dict[str, List[Dict]]:
    """Extract all wrong samples organized by model"""
    
    wrong_by_model = defaultdict(list)
    eval_dir = Path(human_eval_folder)
    
    for json_file in sorted(eval_dir.glob("scenario_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            scenarios = data.get('scenarios', [])
            
            for scenario in scenarios:
                scenario_id = scenario.get('scenario_id', 'unknown')
                domain = scenario.get('domain', 'unknown')
                prompt = scenario.get('prompt', '')
                gold_standard = scenario.get('gold_standard', {})
                
                model_evaluations = scenario.get('model_evaluations', [])
                
                for model_eval in model_evaluations:
                    model_info = model_eval.get('model', {})
                    provider = model_info.get('provider', 'unknown')
                    model = model_info.get('model', 'unknown')
                    model_key = f"{provider}_{model}".replace('/', '_').replace('-', '_')
                    
                    llm_output = model_eval.get('llm_output', {})
                    queries = llm_output.get('queries', [])
                    
                    for query in queries:
                        if query.get('label', '') == 'wrong':
                            wrong_sample = {
                                'scenario_id': scenario_id,
                                'domain': domain,
                                'scenario_prompt': prompt,
                                'gold_standard': gold_standard,
                                'constraint_index': query.get('constraint_index'),
                                'constraint': query.get('constraint', ''),
                                'generated_query': query.get('query', ''),
                                'valid': query.get('valid', False),
                                'pattern': query.get('pattern', ''),
                                'label': 'wrong',
                                'model': {
                                    'provider': provider,
                                    'model': model
                                }
                            }
                            wrong_by_model[model_key].append(wrong_sample)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return wrong_by_model

def save_wrong_samples(wrong_by_model: Dict[str, List[Dict]], output_folder: str = "wrong_samples"):
    """Save wrong samples to separate JSON files by model"""
    
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Create summary
    summary = {
        'total_models': len(wrong_by_model),
        'total_wrong_samples': sum(len(samples) for samples in wrong_by_model.values()),
        'models': {}
    }
    
    for model_key, samples in sorted(wrong_by_model.items()):
        # Save individual model file
        model_file = output_dir / f"{model_key}_wrong_samples.json"
        
        model_data = {
            'model': samples[0]['model'] if samples else {'provider': 'unknown', 'model': 'unknown'},
            'total_wrong_samples': len(samples),
            'wrong_samples': samples
        }
        
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✓ Saved {len(samples)} wrong samples for {model_key}")
        
        # Add to summary
        summary['models'][model_key] = {
            'total_wrong': len(samples),
            'file': str(model_file.name)
        }
    
    # Save summary file
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    return summary

def print_statistics(summary: Dict):
    """Print statistics about wrong samples"""
    print("\n" + "="*80)
    print("WRONG SAMPLES STATISTICS")
    print("="*80)
    
    print(f"\nTotal Models: {summary['total_models']}")
    print(f"Total Wrong Samples: {summary['total_wrong_samples']}")
    
    print("\n" + "-"*80)
    print(f"{'Model':<50} {'Wrong Samples':<15}")
    print("-"*80)
    
    sorted_models = sorted(summary['models'].items(), 
                          key=lambda x: x[1]['total_wrong'], 
                          reverse=True)
    
    for model_key, stats in sorted_models:
        print(f"{model_key:<50} {stats['total_wrong']:<15}")
    
    print("="*80 + "\n")

def main():
    """Main execution"""
    print("Extracting wrong samples from human-eval folder...")
    wrong_by_model = extract_wrong_samples()
    
    print(f"\nFound {sum(len(s) for s in wrong_by_model.values())} wrong samples across {len(wrong_by_model)} models")
    
    print("\nSaving to wrong_samples folder...")
    summary = save_wrong_samples(wrong_by_model)
    
    print_statistics(summary)
    
    print("✅ Extraction complete!")
    print(f"   - Wrong samples organized by model in: wrong_samples/")
    print(f"   - Summary file: wrong_samples/summary.json")

if __name__ == "__main__":
    main()
