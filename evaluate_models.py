#!/usr/bin/env python3
"""
Model Performance Evaluation Script with Precision Metrics
Analyzes all human-eval results to evaluate model effectiveness based on correct/wrong labels
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import statistics

def load_human_eval_data(folder: str = "human-eval") -> List[Dict]:
    """Load all human evaluation JSON files"""
    evaluations = []
    eval_dir = Path(folder)
    
    for json_file in sorted(eval_dir.glob("scenario_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                evaluations.append({
                    'scenario': json_file.stem,
                    'data': data
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return evaluations

def extract_model_metrics(evaluations: List[Dict]) -> Dict[str, Dict]:
    """Extract performance metrics for each model from human evaluations"""
    model_stats = defaultdict(lambda: {
        'total_constraints': 0,
        'correct': 0,
        'wrong': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'labels_distribution': defaultdict(int),
        'patterns_used': defaultdict(int),
        'correct_by_pattern': defaultdict(int),
        'wrong_by_pattern': defaultdict(int),
        'labels_by_pattern': defaultdict(lambda: defaultdict(int)),
        'scenarios': set(),
        'constraints_details': [],
        'constraints_by_label': defaultdict(list)
    })
    
    for evaluation in evaluations:
        scenario_name = evaluation['scenario']
        scenarios = evaluation['data'].get('scenarios', [])
        
        for scenario in scenarios:
            model_evaluations = scenario.get('model_evaluations', [])
            
            for model_eval in model_evaluations:
                model_info = model_eval.get('model', {})
                provider = model_info.get('provider', 'unknown')
                model = model_info.get('model', 'unknown')
                model_key = f"{provider}/{model}"
                
                stats = model_stats[model_key]
                stats['scenarios'].add(scenario_name)
                
                llm_output = model_eval.get('llm_output', {})
                queries = llm_output.get('queries', [])
                
                for query in queries:
                    stats['total_constraints'] += 1
                    
                    label = query.get('label', 'unknown')
                    pattern = query.get('pattern', 'unknown')
                    constraint = query.get('constraint', '')
                    generated_query = query.get('query', '')
                    valid = query.get('valid', False)
                    
                    # Track all labels
                    stats['labels_distribution'][label] += 1
                    stats['patterns_used'][pattern] += 1
                    stats['labels_by_pattern'][pattern][label] += 1
                    
                    if label == 'correct':
                        stats['correct'] += 1
                        stats['correct_by_pattern'][pattern] += 1
                    elif label == 'wrong':
                        stats['wrong'] += 1
                        stats['wrong_by_pattern'][pattern] += 1
                    
                    constraint_detail = {
                        'scenario': scenario_name,
                        'constraint': constraint,
                        'query': generated_query,
                        'label': label,
                        'pattern': pattern,
                        'valid': valid
                    }
                    
                    stats['constraints_details'].append(constraint_detail)
                    stats['constraints_by_label'][label].append(constraint_detail)
    
    return model_stats

def compute_summary_statistics(model_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compute summary statistics for each model"""
    summary = {}
    
    for model_key, stats in model_stats.items():
        total = stats['total_constraints']
        correct = stats['correct']
        wrong = stats['wrong']
        labeled_total = correct + wrong
        
        # Calculate precision (correct / labeled)
        precision = (correct / labeled_total * 100) if labeled_total > 0 else 0
        # Calculate accuracy (correct / total including unlabeled)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Pattern-specific accuracy
        pattern_accuracy = {}
        for pattern in stats['patterns_used'].keys():
            pattern_total = stats['patterns_used'][pattern]
            pattern_correct = stats['correct_by_pattern'].get(pattern, 0)
            pattern_accuracy[pattern] = {
                'total': pattern_total,
                'correct': pattern_correct,
                'wrong': stats['wrong_by_pattern'].get(pattern, 0),
                'accuracy': (pattern_correct / pattern_total * 100) if pattern_total > 0 else 0
            }
        
        summary[model_key] = {
            'model': model_key,
            'total_constraints': total,
            'correct_constraints': correct,
            'wrong_constraints': wrong,
            'accuracy': accuracy,
            'precision': precision,
            'error_rate': (wrong / total * 100) if total > 0 else 0,
            'scenarios_count': len(stats['scenarios']),
            
            # Label distribution
            'labels_distribution': dict(stats['labels_distribution']),
            'labels_by_pattern': {
                pattern: dict(labels) 
                for pattern, labels in stats['labels_by_pattern'].items()
            },
            
            # Pattern analysis
            'patterns_distribution': dict(stats['patterns_used']),
            'pattern_accuracy': pattern_accuracy,
            'most_used_pattern': max(stats['patterns_used'].items(), key=lambda x: x[1])[0] if stats['patterns_used'] else None,
            'best_pattern': max(pattern_accuracy.items(), key=lambda x: x[1]['accuracy'])[0] if pattern_accuracy else None,
            'worst_pattern': min(pattern_accuracy.items(), key=lambda x: x[1]['accuracy'])[0] if pattern_accuracy else None,
            
            # Detailed breakdown
            'correct_by_pattern': dict(stats['correct_by_pattern']),
            'wrong_by_pattern': dict(stats['wrong_by_pattern']),
            
            # Constraints grouped by label
            'constraints_by_label': {
                label: samples for label, samples in stats['constraints_by_label'].items()
            }
        }
    
    return summary

def create_benchmark_json(summary: Dict[str, Dict], output_file: str = "model_benchmarks.json"):
    """Create comprehensive benchmark JSON file"""
    
    # Sort models by precision
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['precision'], reverse=True)
    
    benchmark_data = {
        'evaluation_date': '2026-01-29',
        'evaluation_source': 'human-eval folder',
        'total_models_evaluated': len(summary),
        'models_ranked_by_precision': [
            {
                'rank': idx + 1,
                'model': model_key,
                'precision': stats['precision'],
                'accuracy': stats['accuracy'],
                'correct': stats['correct_constraints'],
                'wrong': stats['wrong_constraints'],
                'total': stats['total_constraints']
            }
            for idx, (model_key, stats) in enumerate(sorted_models)
        ],
        'detailed_metrics': {
            model_key: stats for model_key, stats in summary.items()
        },
        'comparative_analysis': {
            'best_precision': {
                'model': sorted_models[0][0],
                'precision': sorted_models[0][1]['precision'],
                'accuracy': sorted_models[0][1]['accuracy'],
                'correct': sorted_models[0][1]['correct_constraints'],
                'total': sorted_models[0][1]['total_constraints']
            } if sorted_models else None,
            'worst_precision': {
                'model': sorted_models[-1][0],
                'precision': sorted_models[-1][1]['precision'],
                'accuracy': sorted_models[-1][1]['accuracy'],
                'correct': sorted_models[-1][1]['correct_constraints'],
                'total': sorted_models[-1][1]['total_constraints']
            } if sorted_models else None,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"✓ Benchmark JSON saved to {output_file}")
    return benchmark_data

def print_summary_report(benchmark_data: Dict):
    """Print a text summary report"""
    print("\n" + "="*90)
    print("MODEL PERFORMANCE EVALUATION REPORT (Human Evaluation)")
    print("="*90)
    
    print(f"\nEvaluation Date: {benchmark_data['evaluation_date']}")
    print(f"Evaluation Source: {benchmark_data['evaluation_source']}")
    print(f"Total Models Evaluated: {benchmark_data['total_models_evaluated']}")
    
    print("\n" + "-"*90)
    print("MODELS RANKED BY PRECISION:")
    print("-"*90)
    print(f"{'Rank':<6} {'Model':<35} {'Precision':<12} {'Accuracy':<12} {'Correct':<10}")
    print("-"*90)
    
    for rank_info in benchmark_data['models_ranked_by_precision']:
        print(f"{rank_info['rank']:<6} {rank_info['model']:<35} "
              f"{rank_info['precision']:>10.2f}%  {rank_info['accuracy']:>10.2f}%  {rank_info['correct']:<10}")
    
    print("\n" + "-"*90)
    print("TOP PERFORMERS:")
    print("-"*90)
    
    comp = benchmark_data['comparative_analysis']
    best = comp['best_precision']
    worst = comp['worst_precision']
    
    print(f"\nBest Precision:")
    print(f"  Model:    {best['model']}")
    print(f"  Precision: {best['precision']:.2f}%")
    print(f"  Accuracy: {best['accuracy']:.2f}%")
    print(f"  Correct:  {best['correct']}/{best['total']}")
    
    print(f"\nWorst Precision:")
    print(f"  Model:    {worst['model']}")
    print(f"  Precision: {worst['precision']:.2f}%")
    print(f"  Accuracy: {worst['accuracy']:.2f}%")
    print(f"  Correct:  {worst['correct']}/{worst['total']}")
    
    print("\n" + "="*90 + "\n")

def main():
    """Main execution function"""
    print("Loading human evaluation data...")
    evaluations = load_human_eval_data()
    print(f"✓ Loaded {len(evaluations)} scenario evaluations")
    
    print("\nExtracting model metrics from human evaluations...")
    model_stats = extract_model_metrics(evaluations)
    print(f"✓ Analyzed {len(model_stats)} models")
    
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(model_stats)
    
    print("\nCreating benchmark JSON with Precision metrics...")
    benchmark_data = create_benchmark_json(summary)
    
    print_summary_report(benchmark_data)
    
    print("✅ Evaluation complete!")
    print(f"   - Benchmark data: model_benchmarks.json")

if __name__ == "__main__":
    main()
