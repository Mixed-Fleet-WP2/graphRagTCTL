#!/usr/bin/env python3
"""
Model Performance Evaluation Script
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
            'accuracy': (correct / total * 100) if total > 0 else 0,
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
    
    # Sort models by accuracy
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    benchmark_data = {
        'evaluation_date': '2026-01-27',
        'evaluation_source': 'human-eval folder',
        'total_models_evaluated': len(summary),
        'models_ranked_by_accuracy': [
            {
                'rank': idx + 1,
                'model': model_key,
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
            'best_accuracy': {
                'model': sorted_models[0][0],
                'accuracy': sorted_models[0][1]['accuracy'],
                'correct': sorted_models[0][1]['correct_constraints'],
                'total': sorted_models[0][1]['total_constraints']
            } if sorted_models else None,
            'worst_accuracy': {
                'model': sorted_models[-1][0],
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

def create_visualizations(benchmark_data: Dict):
    """Create visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping visualizations...")
        return
    
    detailed_metrics = benchmark_data['detailed_metrics']
    models = list(detailed_metrics.keys())
    
    # Shorten model names for better display
    model_labels = [m.split('/')[-1][:20] for m in models]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison (Human Evaluation)', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    ax1 = axes[0, 0]
    accuracy = [detailed_metrics[m]['accuracy'] for m in models]
    colors = plt.cm.RdYlGn(np.array(accuracy) / 100)
    bars1 = ax1.barh(model_labels, accuracy, color=colors)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Accuracy by Model')
    ax1.set_xlim(0, 100)
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    for i, v in enumerate(accuracy):
        ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 2. Label Distribution Stacked Bar
    ax2 = axes[0, 1]
    all_labels = set()
    for m in models:
        all_labels.update(detailed_metrics[m]['labels_distribution'].keys())
    
    label_colors = {'correct': 'green', 'wrong': 'red', 'unknown': 'gray'}
    
    if all_labels:
        x_pos = np.arange(len(models))
        bottom = np.zeros(len(models))
        
        for label in sorted(all_labels):
            counts = [detailed_metrics[m]['labels_distribution'].get(label, 0) for m in models]
            color = label_colors.get(label, 'blue')
            ax2.bar(x_pos, counts, bottom=bottom, label=label, color=color, alpha=0.7)
            bottom += counts
        
        ax2.set_ylabel('Count')
        ax2.set_title('Label Distribution by Model')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Error Rate
    ax3 = axes[0, 2]
    error_rates = [detailed_metrics[m]['error_rate'] for m in models]
    colors_err = plt.cm.Reds(np.array(error_rates) / max(error_rates) if max(error_rates) > 0 else [1])
    ax3.barh(model_labels, error_rates, color=colors_err)
    ax3.set_xlabel('Error Rate (%)')
    ax3.set_title('Error Rate by Model')
    for i, v in enumerate(error_rates):
        ax3.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Pattern Usage Distribution
    ax4 = axes[1, 0]
    all_patterns = set()
    for m in models:
        all_patterns.update(detailed_metrics[m]['patterns_distribution'].keys())
    
    if all_patterns:
        pattern_data = []
        for pattern in sorted(all_patterns):
            counts = [detailed_metrics[m]['patterns_distribution'].get(pattern, 0) for m in models]
            pattern_data.append(counts)
        
        x_pos = np.arange(len(models))
        width = 0.8 / len(all_patterns) if all_patterns else 0.1
        
        for i, (pattern, counts) in enumerate(zip(sorted(all_patterns), pattern_data)):
            offset = (i - len(all_patterns)/2) * width
            ax4.bar(x_pos + offset, counts, width, label=pattern[:15], alpha=0.8)
        
        ax4.set_ylabel('Count')
        ax4.set_title('Pattern Usage Distribution')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Pattern-Specific Accuracy Heatmap
    ax5 = axes[1, 1]
    if all_patterns:
        accuracy_matrix = []
        for m in models:
            row = []
            for pattern in sorted(all_patterns):
                pattern_acc = detailed_metrics[m]['pattern_accuracy'].get(pattern, {}).get('accuracy', 0)
                row.append(pattern_acc)
            accuracy_matrix.append(row)
        
        im = ax5.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax5.set_xticks(np.arange(len(all_patterns)))
        ax5.set_yticks(np.arange(len(models)))
        ax5.set_xticklabels([p[:15] for p in sorted(all_patterns)], rotation=45, ha='right', fontsize=7)
        ax5.set_yticklabels(model_labels, fontsize=8)
        ax5.set_title('Pattern-Specific Accuracy Heatmap (%)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Accuracy %', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(all_patterns)):
                text = ax5.text(j, i, f'{accuracy_matrix[i][j]:.0f}',
                               ha="center", va="center", color="black", fontsize=7)
    
    # 6. Overall Performance Score
    ax6 = axes[1, 2]
    # Composite score based on accuracy and coverage
    performance_scores = []
    for m in models:
        acc = detailed_metrics[m]['accuracy']
        total = detailed_metrics[m]['total_constraints']
        # Weight by both accuracy and number of scenarios covered
        score = acc * (total / 200)  # Normalize by approximate max constraints
        performance_scores.append(score)
    
    colors_perf = plt.cm.viridis(np.array(performance_scores) / max(performance_scores) if performance_scores else [1])
    ax6.barh(model_labels, performance_scores, color=colors_perf)
    ax6.set_xlabel('Performance Score')
    ax6.set_title('Overall Performance Score\n(Accuracy × Coverage)')
    for i, v in enumerate(performance_scores):
        ax6.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'model_performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_file}")
    
    # Create additional detailed plots
    create_detailed_plots(detailed_metrics, models, model_labels)
    
    plt.show()

def create_detailed_plots(detailed_metrics, models, model_labels):
    """Create additional detailed analysis plots"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Pattern and Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Correct vs Wrong by Pattern
    ax1 = axes[0, 0]
    all_patterns = set()
    for m in models:
        all_patterns.update(detailed_metrics[m]['correct_by_pattern'].keys())
        all_patterns.update(detailed_metrics[m]['wrong_by_pattern'].keys())
    
    if all_patterns:
        x_pos = np.arange(len(all_patterns))
        width = 0.8 / len(models) if models else 0.1
        
        for i, m in enumerate(models):
            correct = [detailed_metrics[m]['correct_by_pattern'].get(p, 0) for p in sorted(all_patterns)]
            offset = (i - len(models)/2) * width
            ax1.bar(x_pos + offset, correct, width, label=model_labels[i], alpha=0.8)
        
        ax1.set_ylabel('Correct Count')
        ax1.set_title('Correct Queries by Pattern')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([p[:15] for p in sorted(all_patterns)], rotation=45, ha='right', fontsize=8)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Wrong queries by pattern
    ax2 = axes[0, 1]
    if all_patterns:
        for i, m in enumerate(models):
            wrong = [detailed_metrics[m]['wrong_by_pattern'].get(p, 0) for p in sorted(all_patterns)]
            offset = (i - len(models)/2) * width
            ax2.bar(x_pos + offset, wrong, width, label=model_labels[i], alpha=0.8)
        
        ax2.set_ylabel('Wrong Count')
        ax2.set_title('Wrong Queries by Pattern')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([p[:15] for p in sorted(all_patterns)], rotation=45, ha='right', fontsize=8)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Model Ranking
    ax3 = axes[1, 0]
    sorted_models = sorted(enumerate(models), key=lambda x: detailed_metrics[x[1]]['accuracy'], reverse=True)
    ranks = [i+1 for i in range(len(models))]
    accuracies = [detailed_metrics[models[i]]['accuracy'] for i, _ in sorted_models]
    sorted_labels = [model_labels[i] for i, _ in sorted_models]
    
    colors_rank = plt.cm.RdYlGn(np.array(accuracies) / 100)
    ax3.barh(ranks, accuracies, color=colors_rank)
    ax3.set_yticks(ranks)
    ax3.set_yticklabels([f"{r}. {l}" for r, l in zip(ranks, sorted_labels)], fontsize=9)
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Model Ranking by Accuracy')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    for i, (rank, acc) in enumerate(zip(ranks, accuracies)):
        ax3.text(acc + 1, rank, f'{acc:.1f}%', va='center', fontsize=9)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    headers = ['Model', 'Accuracy', 'Correct', 'Wrong', 'Total']
    
    for m in sorted(models, key=lambda x: detailed_metrics[x]['accuracy'], reverse=True):
        label = m.split('/')[-1][:25]
        acc = f"{detailed_metrics[m]['accuracy']:.1f}%"
        correct = detailed_metrics[m]['correct_constraints']
        wrong = detailed_metrics[m]['wrong_constraints']
        total = detailed_metrics[m]['total_constraints']
        table_data.append([label, acc, correct, wrong, total])
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color code the accuracy column
    for i in range(1, len(table_data) + 1):
        acc_val = float(table_data[i-1][1].rstrip('%'))
        color = plt.cm.RdYlGn(acc_val / 100)
        table[(i, 1)].set_facecolor(color)
    
    ax4.set_title('Summary Statistics', fontsize=12, pad=20)
    
    plt.tight_layout()
    output_file = 'detailed_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed analysis saved to {output_file}")

def print_summary_report(benchmark_data: Dict):
    """Print a text summary report"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE EVALUATION REPORT (Human Evaluation)")
    print("="*80)
    
    print(f"\nEvaluation Date: {benchmark_data['evaluation_date']}")
    print(f"Evaluation Source: {benchmark_data['evaluation_source']}")
    print(f"Total Models Evaluated: {benchmark_data['total_models_evaluated']}")
    
    print("\n" + "-"*80)
    print("MODELS RANKED BY ACCURACY:")
    print("-"*80)
    print(f"{'Rank':<6} {'Model':<40} {'Accuracy':<10} {'Correct':<8} {'Wrong':<8} {'Total':<8}")
    print("-"*80)
    
    for rank_info in benchmark_data['models_ranked_by_accuracy']:
        print(f"{rank_info['rank']:<6} {rank_info['model']:<40} "
              f"{rank_info['accuracy']:>7.2f}%  {rank_info['correct']:<8} "
              f"{rank_info['wrong']:<8} {rank_info['total']:<8}")
    
    print("\n" + "-"*80)
    print("TOP PERFORMERS:")
    print("-"*80)
    
    comp = benchmark_data['comparative_analysis']
    best = comp['best_accuracy']
    worst = comp['worst_accuracy']
    
    print(f"\nBest Accuracy:")
    print(f"  Model:    {best['model']}")
    print(f"  Accuracy: {best['accuracy']:.2f}%")
    print(f"  Correct:  {best['correct']}/{best['total']}")
    
    print(f"\nWorst Accuracy:")
    print(f"  Model:    {worst['model']}")
    print(f"  Accuracy: {worst['accuracy']:.2f}%")
    print(f"  Correct:  {worst['correct']}/{worst['total']}")
    
    # Label distribution summary
    print("\n" + "-"*80)
    print("LABEL DISTRIBUTION ACROSS ALL MODELS:")
    print("-"*80)
    
    detailed = benchmark_data['detailed_metrics']
    all_labels = set()
    for model_stats in detailed.values():
        all_labels.update(model_stats['labels_distribution'].keys())
    
    label_totals = defaultdict(int)
    for model_stats in detailed.values():
        for label, count in model_stats['labels_distribution'].items():
            label_totals[label] += count
    
    total_samples = sum(label_totals.values())
    print(f"\n{'Label':<20} {'Count':<10} {'Percentage':<10}")
    print("-"*40)
    for label in sorted(label_totals.keys()):
        count = label_totals[label]
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{label:<20} {count:<10} {pct:>6.2f}%")
    
    # Pattern analysis
    print("\n" + "-"*80)
    print("PATTERN ANALYSIS (Top 3 Models):")
    print("-"*80)
    
    for model_key in sorted(detailed.keys(), key=lambda x: detailed[x]['accuracy'], reverse=True)[:3]:
        stats = detailed[model_key]
        print(f"\n{model_key}:")
        print(f"  Most used pattern: {stats['most_used_pattern']}")
        print(f"  Best pattern:      {stats['best_pattern']}")
        if stats['worst_pattern']:
            print(f"  Worst pattern:     {stats['worst_pattern']}")
        
        print(f"  Pattern breakdown:")
        for pattern, acc_info in sorted(stats['pattern_accuracy'].items(), 
                                       key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"    - {pattern:<25}: {acc_info['accuracy']:>6.1f}% "
                  f"({acc_info['correct']}/{acc_info['total']})")
    
    print("\n" + "="*80 + "\n")

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
    
    print("\nCreating benchmark JSON...")
    benchmark_data = create_benchmark_json(summary)
    
    print("\nGenerating visualizations...")
    create_visualizations(benchmark_data)
    
    print_summary_report(benchmark_data)
    
    print("\n✅ Evaluation complete!")
    print(f"   - Benchmark data: model_benchmarks.json")
    print(f"   - Main plots: model_performance_comparison.png")
    print(f"   - Detailed analysis: detailed_analysis.png")

if __name__ == "__main__":
    main()
