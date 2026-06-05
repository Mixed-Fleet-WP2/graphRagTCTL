#!/usr/bin/env python3
"""
Compare performance between prompting and GraphRAG approaches
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_benchmark_data():
    """Load data from both benchmark JSON files"""
    comparaison_dir = Path("comparaison")
    
    # Load prompting-only results
    with open(comparaison_dir / "prompting_only_bench.json", 'r') as f:
        prompting_data = json.load(f)
    
    # Load GraphRAG results  
    with open(comparaison_dir / "grapghRag_bench.json", 'r') as f:
        graphrag_data = json.load(f)
    
    return prompting_data, graphrag_data

def normalize_model_name(model_name: str) -> str:
    """Normalize model names so different separators and providers match across benchmark files."""
    name = (model_name or "").strip().lower()
    name = name.replace("::", "/")
    name = name.replace("deepseek-r1-14b", "deepseek-r1:14b")
    name = name.replace("deepseek-r1:14b", "deepseek-r1:14b")
    # Use the model slug only, ignoring provider prefixes like openai/, ollama/, deepseek/.
    if "/" in name:
        name = name.split("/")[-1]
    return name


def display_model_name(model_name: str) -> str:
    """Return a compact label for chart axes."""
    return normalize_model_name(model_name)

def extract_model_performance(data):
    """Extract model names and their accuracy scores"""
    models = []
    accuracy_scores = []
    
    for model_data in data["models_ranked_by_precision"]:
        # Clean up model names for better display
        models.append(display_model_name(model_data["model"]))
        accuracy_scores.append(model_data["accuracy"])
    
    return models, accuracy_scores

def extract_model_map(data):
    """Return normalized model keys mapped to accuracy scores and display labels."""
    model_map = {}
    label_map = {}

    for model_data in data["models_ranked_by_precision"]:
        key = normalize_model_name(model_data["model"])
        model_map[key] = model_data["accuracy"]
        label_map[key] = display_model_name(model_data["model"])

    return model_map, label_map

def create_comparison_plots():
    """Create comprehensive comparison visualizations"""
    # Load data
    prompting_data, graphrag_data = load_benchmark_data()
    
    # Extract performance data
    prompting_models, prompting_scores = extract_model_performance(prompting_data)
    graphrag_models, graphrag_scores = extract_model_performance(graphrag_data)
    
    # Create a mapping of models to scores for both approaches
    prompting_dict = dict(zip(prompting_models, prompting_scores))
    graphrag_dict = dict(zip(graphrag_models, graphrag_scores))
    
    # Find common models (should be all models, but let's be safe)
    common_models = []
    prompting_common = []
    graphrag_common = []
    
    for model in prompting_models:
        if model in graphrag_dict:
            common_models.append(model)
            prompting_common.append(prompting_dict[model])
            graphrag_common.append(graphrag_dict[model])
    
    # Sort by GraphRAG performance for better visualization
    sorted_indices = np.argsort(graphrag_common)[::-1]
    common_models = [common_models[i] for i in sorted_indices]
    prompting_common = [prompting_common[i] for i in sorted_indices]
    graphrag_common = [graphrag_common[i] for i in sorted_indices]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Side-by-side bar comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(common_models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, prompting_common, width, label='Few Shot Prompting', 
                    color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, graphrag_common, width, label='GraphRAG', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy Comparison: Few Shot Prompting vs GraphRAG', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Improvement/Difference plot
    ax2 = plt.subplot(2, 3, 2)
    differences = [g - p for g, p in zip(graphrag_common, prompting_common)]
    colors = ['green' if d > 0 else 'red' for d in differences]
    
    bars = ax2.bar(range(len(common_models)), differences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Accuracy Difference (GraphRAG - Few Shot Prompting)', fontweight='bold')
    ax2.set_title('Performance Improvement with GraphRAG', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(common_models)))
    ax2.set_xticklabels(common_models, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
                f'{diff:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # 3. Scatter plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(prompting_common, graphrag_common, s=100, alpha=0.7, c='purple', edgecolors='black')
    
    # Add diagonal line (y=x)
    min_val = min(min(prompting_common), min(graphrag_common))
    max_val = max(max(prompting_common), max(graphrag_common))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Performance')
    
    ax3.set_xlabel('Few Shot Prompting Accuracy (%)', fontweight='bold')
    ax3.set_ylabel('GraphRAG Accuracy (%)', fontweight='bold')
    ax3.set_title('Accuracy Correlation', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add model labels to points
    for i, model in enumerate(common_models):
        ax3.annotate(model, (prompting_common[i], graphrag_common[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # 4. Top performers comparison
    ax4 = plt.subplot(2, 3, 4)
    top_n = 5
    top_models = common_models[:top_n]
    top_prompting = prompting_common[:top_n]
    top_graphrag = graphrag_common[:top_n]
    
    x = np.arange(len(top_models))
    bars1 = ax4.bar(x - width/2, top_prompting, width, label='Few Shot Prompting', 
                    color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    bars2 = ax4.bar(x + width/2, top_graphrag, width, label='GraphRAG', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    
    ax4.set_xlabel('Top 5 Models', fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontweight='bold')
    ax4.set_title('Top 5 Models Comparison', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary statistics
    ax5 = plt.subplot(2, 3, 5)
    stats_labels = ['Average', 'Median', 'Max', 'Min', 'Std Dev']
    prompting_stats = [
        np.mean(prompting_common),
        np.median(prompting_common), 
        np.max(prompting_common),
        np.min(prompting_common),
        np.std(prompting_common)
    ]
    graphrag_stats = [
        np.mean(graphrag_common),
        np.median(graphrag_common),
        np.max(graphrag_common), 
        np.min(graphrag_common),
        np.std(graphrag_common)
    ]
    
    x = np.arange(len(stats_labels))
    bars1 = ax5.bar(x - width/2, prompting_stats, width, label='Few Shot Prompting', 
                    color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    bars2 = ax5.bar(x + width/2, graphrag_stats, width, label='GraphRAG', 
                    color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    
    ax5.set_xlabel('Statistics', fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontweight='bold')
    ax5.set_title('Statistical Summary', fontweight='bold', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(stats_labels, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Model ranking comparison
    ax6 = plt.subplot(2, 3, 6)
    
    # Create ranking data - need to get actual rankings from original data
    prompting_ranks = []
    graphrag_ranks = []
    
    # Get actual rankings for each model
    for model in common_models:
        # Find ranking in prompting-only (1-based)
        prompting_rank = prompting_models.index(model) + 1
        prompting_ranks.append(prompting_rank)
        
        # Find ranking in GraphRAG (1-based) 
        graphrag_rank = graphrag_models.index(model) + 1
        graphrag_ranks.append(graphrag_rank)
    
    # Plot ranking comparison
    for i, model in enumerate(common_models):
        ax6.plot([1, 2], [prompting_ranks[i], graphrag_ranks[i]], 'o-', alpha=0.7, linewidth=2, markersize=8)
        ax6.text(0.95, prompting_ranks[i], model, ha='right', va='center', fontsize=8)
        ax6.text(2.05, graphrag_ranks[i], model, ha='left', va='center', fontsize=8)
    
    ax6.set_xlim(0.5, 2.5)
    ax6.set_ylim(0.5, len(common_models) + 0.5)
    ax6.set_xticks([1, 2])
    ax6.set_xticklabels(['Few Shot Prompting\nRanking', 'GraphRAG\nRanking'], fontweight='bold')
    ax6.set_ylabel('Rank (1=Best)', fontweight='bold')
    ax6.set_title('Ranking Changes', fontweight='bold', fontsize=14)
    ax6.invert_yaxis()  # Invert so rank 1 is at top
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('approach_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison visualization saved to approach_comparison.png")

    # Also regenerate the focused vertical comparison chart.
    prompting_map, prompting_labels = extract_model_map(prompting_data)
    graphrag_map, graphrag_labels = extract_model_map(graphrag_data)

    common_keys = [m for m in prompting_map.keys() if m in graphrag_map]
    common_keys = sorted(common_keys, key=lambda m: graphrag_map[m], reverse=True)
    prompting_vertical = [prompting_map[m] for m in common_keys]
    graphrag_vertical = [graphrag_map[m] for m in common_keys]
    xtick_labels = [graphrag_labels.get(m, prompting_labels.get(m, m)) for m in common_keys]

    fig2, axv = plt.subplots(figsize=(14, 7))
    x2 = np.arange(len(common_keys))
    width2 = 0.36
    bars1 = axv.bar(x2 - width2/2, prompting_vertical, width2, label='Few-shot Prompting', color='#4C78A8')
    bars2 = axv.bar(x2 + width2/2, graphrag_vertical, width2, label='GraphRAG', color='#F58518')

    axv.set_title('Few-shot Prompting vs GraphRAG Performance by Model', fontsize=14, pad=12)
    axv.set_ylabel('Accuracy (%)')
    axv.set_xticks(x2)
    axv.set_xticklabels(xtick_labels, rotation=35, ha='right', fontsize=9)
    axv.set_ylim(0, 100)
    axv.grid(axis='y', linestyle='--', alpha=0.35)
    axv.legend(loc='upper left')

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            axv.annotate(f'{h:.1f}', (b.get_x() + b.get_width()/2, h),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8,
                         clip_on=False,
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'))

    avg_p = float(np.mean(prompting_vertical)) if prompting_vertical else 0.0
    avg_g = float(np.mean(graphrag_vertical)) if graphrag_vertical else 0.0
    summary = f'Avg Few-shot: {avg_p:.2f}%\nAvg GraphRAG: {avg_g:.2f}%\nImprovement: +{(avg_g-avg_p):.2f} pts'
    axv.text(0.985, 0.985, summary, transform=axv.transAxes, ha='right', va='top',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig('comparaison/fewshot_vs_graphrag_vertical.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison visualization saved to comparaison/fewshot_vs_graphrag_vertical.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("APPROACH COMPARISON SUMMARY")
    print("="*60)
    print(f"Models evaluated: {len(common_models)}")
    print(f"\nFew Shot Prompting Approach:")
    print(f"  Average accuracy: {np.mean(prompting_common):.2f}%")
    print(f"  Best model: {common_models[prompting_common.index(max(prompting_common))]} ({max(prompting_common):.2f}%)")
    print(f"  Worst model: {common_models[prompting_common.index(min(prompting_common))]} ({min(prompting_common):.2f}%)")
    
    print(f"\nGraphRAG Approach:")
    print(f"  Average accuracy: {np.mean(graphrag_common):.2f}%") 
    print(f"  Best model: {common_models[graphrag_common.index(max(graphrag_common))]} ({max(graphrag_common):.2f}%)")
    print(f"  Worst model: {common_models[graphrag_common.index(min(graphrag_common))]} ({min(graphrag_common):.2f}%)")
    
    avg_improvement = np.mean(graphrag_common) - np.mean(prompting_common)
    print(f"\nOverall Improvement:")
    print(f"  Average improvement: {avg_improvement:+.2f}%")
    print(f"  Models improved: {sum(1 for d in differences if d > 0)}/{len(differences)}")
    print(f"  Models degraded: {sum(1 for d in differences if d < 0)}/{len(differences)}")
    print(f"  Largest improvement: {max(differences):+.2f}%")
    print(f"  Largest degradation: {min(differences):+.2f}%")

if __name__ == "__main__":
    create_comparison_plots()