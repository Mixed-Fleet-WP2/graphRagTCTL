#!/usr/bin/env python3
"""
Analyze wrong samples: classify errors as syntax (valid=false) or semantic (valid=true).
Returns counts and percentages.
"""

import json
import os


def count_total_generated_properties(dataset_dir='dataset'):
    """Count total reference properties in dataset (denominator per model)."""
    total = 0
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    for file in sorted(files):
        path = os.path.join(dataset_dir, file)
        with open(path, 'r') as f:
            data = json.load(f)
        total += len(data.get('reference_properties', []))
    return total

def analyze_error_types():
    """Analyze all wrong_samples files and classify errors with percentages."""
    wrong_samples_dir = 'wrong_samples'
    files = [f for f in os.listdir(wrong_samples_dir) if f.endswith('_wrong_samples.json')]
    denominator_per_model = count_total_generated_properties('dataset')

    analysis = {
        'denominator_per_model': denominator_per_model,
        'total_models': 0,
        'total_generated_all_models': 0,
        'total_syntax_errors': 0,
        'total_semantic_errors': 0,
        'total_wrong': 0,
        'syntax_percentage': 0.0,
        'semantic_percentage': 0.0,
        'total_syntax_error_percentage_overall': 0.0,
        'total_semantics_error_percentage_overall': 0.0,
        'total_error_percentage_overall': 0.0,
        'by_model': {},
    }

    for file in sorted(files):
        path = os.path.join(wrong_samples_dir, file)
        with open(path, 'r') as f:
            data = json.load(f)
        
        syntax_count = 0
        semantic_count = 0
        
        for sample in data.get('wrong_samples', []):
            # Classification: valid=false → syntax error, valid=true → semantic error
            if sample.get('valid') == False:
                syntax_count += 1
            elif sample.get('valid') == True:
                semantic_count += 1
        
        total_model = syntax_count + semantic_count
        syntax_pct = (syntax_count / total_model * 100) if total_model > 0 else 0.0
        semantic_pct = (semantic_count / total_model * 100) if total_model > 0 else 0.0
        syntax_overall_pct = (syntax_count / denominator_per_model * 100) if denominator_per_model > 0 else 0.0
        semantic_overall_pct = (semantic_count / denominator_per_model * 100) if denominator_per_model > 0 else 0.0
        total_overall_pct = (total_model / denominator_per_model * 100) if denominator_per_model > 0 else 0.0
        
        model_name = data['model']['provider'] + '/' + data['model']['model']
        analysis['by_model'][model_name] = {
            'syntax_errors': syntax_count,
            'semantic_errors': semantic_count,
            'total': total_model,
            'syntax_percentage': round(syntax_pct, 2),
            'semantic_percentage': round(semantic_pct, 2),
            'syntax_error_percentage_overall': round(syntax_overall_pct, 2),
            'semantics_error_percentage_overall': round(semantic_overall_pct, 2),
            'total_error_percentage_overall': round(total_overall_pct, 2),
        }
        analysis['total_syntax_errors'] += syntax_count
        analysis['total_semantic_errors'] += semantic_count
        analysis['total_wrong'] += total_model

    analysis['total_models'] = len(analysis['by_model'])
    analysis['total_generated_all_models'] = analysis['total_models'] * denominator_per_model

    # Calculate overall percentages
    if analysis['total_wrong'] > 0:
        analysis['syntax_percentage'] = round(analysis['total_syntax_errors'] / analysis['total_wrong'] * 100, 2)
        analysis['semantic_percentage'] = round(analysis['total_semantic_errors'] / analysis['total_wrong'] * 100, 2)

    if analysis['total_generated_all_models'] > 0:
        analysis['total_syntax_error_percentage_overall'] = round(
            analysis['total_syntax_errors'] / analysis['total_generated_all_models'] * 100, 2
        )
        analysis['total_semantics_error_percentage_overall'] = round(
            analysis['total_semantic_errors'] / analysis['total_generated_all_models'] * 100, 2
        )
        analysis['total_error_percentage_overall'] = round(
            analysis['total_wrong'] / analysis['total_generated_all_models'] * 100, 2
        )

    return analysis


if __name__ == '__main__':
    result = analyze_error_types()
    print(json.dumps(result, indent=2))
    
    # Save to file
    with open('error_types_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n✓ Analysis saved to error_types_analysis.json")
