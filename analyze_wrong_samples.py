#!/usr/bin/env python3
"""
Analyze wrong samples to identify patterns and provide improvement recommendations
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import re

def load_wrong_samples(folder: str = "wrong_samples"):
    """Load all wrong samples from the wrong_samples folder"""
    wrong_samples = {}
    summary = None
    
    folder_path = Path(folder)
    
    for json_file in folder_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if json_file.stem == 'summary':
            summary = data
        else:
            # Handle both dict and string model keys
            model_data = data.get('model', 'unknown')
            if isinstance(model_data, dict):
                provider = model_data.get('provider', 'unknown')
                model_name = model_data.get('model', 'unknown')
                model_key = f"{provider}/{model_name}"
            else:
                model_key = model_data
            wrong_samples[model_key] = data.get('wrong_samples', [])
    
    return wrong_samples, summary

def analyze_error_patterns(wrong_samples: dict):
    """Analyze common error patterns across all wrong samples"""
    
    analysis = {
        'pattern_errors': defaultdict(lambda: defaultdict(int)),
        'model_pattern_errors': defaultdict(lambda: defaultdict(int)),
        'common_constraint_issues': defaultdict(list),
        'query_characteristics': defaultdict(list),
        'scenario_hotspots': defaultdict(int),
        'pattern_difficulty': defaultdict(lambda: {'total': 0, 'errors': 0}),
    }
    
    all_samples = []
    
    for model_key, samples in wrong_samples.items():
        for sample in samples:
            all_samples.append({**sample, 'model': model_key})
            
            pattern = sample.get('pattern', 'unknown')
            scenario = sample.get('scenario', 'unknown')
            constraint = sample.get('constraint', '')
            # Try both 'query' and 'generated_query' field names
            query = sample.get('generated_query', sample.get('query', ''))
            
            # Track pattern-specific errors
            analysis['pattern_errors'][pattern]['count'] += 1
            analysis['model_pattern_errors'][model_key][pattern] += 1
            
            # Track scenario difficulty
            analysis['scenario_hotspots'][scenario] += 1
            
            # Track pattern difficulty
            analysis['pattern_difficulty'][pattern]['errors'] += 1
            
            # Collect constraint issues
            analysis['common_constraint_issues'][pattern].append({
                'constraint': constraint,
                'query': query,
                'scenario': scenario,
                'model': model_key
            })
            
            # Analyze query characteristics
            query_analysis = {
                'pattern': pattern,
                'model': model_key,
                'query_length': len(query),
                'has_temporal': any(word in query.lower() for word in ['time', 'clock', 'deadline', 'eventually', 'always', 'until']),
                'has_logic': any(word in query.lower() for word in ['and', 'or', 'not', 'imply', '&&', '||', '!']),
                'has_variables': bool(re.search(r'\b[a-zA-Z_]\w*\b', query)),
                'has_operators': any(op in query for op in ['==', '!=', '<', '>', '<=', '>=']),
                'complexity_score': len(query.split()) + query.count('(') + query.count('&&') + query.count('||')
            }
            analysis['query_characteristics'][pattern].append(query_analysis)
    
    return analysis, all_samples

def identify_common_mistakes(all_samples: list):
    """Identify common mistakes across samples"""
    
    mistakes = {
        'syntax_patterns': defaultdict(int),
        'semantic_issues': defaultdict(int),
        'temporal_logic_errors': defaultdict(int),
        'variable_issues': defaultdict(int),
    }
    
    for sample in all_samples:
        # Try both 'query' and 'generated_query' field names
        query = sample.get('generated_query', sample.get('query', ''))
        constraint = sample.get('constraint', '')
        pattern = sample.get('pattern', 'unknown')
        
        # Check for common syntax issues
        if query.count('(') != query.count(')'):
            mistakes['syntax_patterns']['unbalanced_parentheses'] += 1
        
        if '  ' in query:
            mistakes['syntax_patterns']['extra_whitespace'] += 1
        
        if query.strip() == '':
            mistakes['syntax_patterns']['empty_query'] += 1
        
        # Check for temporal logic issues
        if 'E<>' in query or 'A[]' in query:
            mistakes['temporal_logic_errors']['ctl_instead_of_tctl'] += 1
        
        if any(word in constraint.lower() for word in ['time', 'deadline', 'within', 'before', 'after']) and 'clock' not in query.lower():
            mistakes['temporal_logic_errors']['missing_clock_constraints'] += 1
        
        # Check for variable issues
        constraint_vars = set(re.findall(r'\b[a-z_]\w*\b', constraint.lower()))
        query_vars = set(re.findall(r'\b[a-z_]\w*\b', query.lower()))
        
        if constraint_vars and not query_vars.intersection(constraint_vars):
            mistakes['variable_issues']['variable_mismatch'] += 1
        
        # Semantic issues
        if 'always' in constraint.lower() and 'A[]' not in query and 'A<>' not in query:
            mistakes['semantic_issues']['always_not_captured'] += 1
        
        if 'eventually' in constraint.lower() and 'E<>' not in query and 'E[]' not in query:
            mistakes['semantic_issues']['eventually_not_captured'] += 1
        
        if ('never' in constraint.lower() or 'not' in constraint.lower()) and '!' not in query and 'not' not in query.lower():
            mistakes['semantic_issues']['negation_missing'] += 1
    
    return mistakes

def generate_recommendations(analysis: dict, mistakes: dict, wrong_samples: dict):
    """Generate actionable recommendations for improvement"""
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'model_specific': defaultdict(list),
        'pattern_specific': defaultdict(list),
    }
    
    # Analyze pattern difficulty
    pattern_error_rates = {}
    for pattern, stats in analysis['pattern_difficulty'].items():
        if stats['errors'] > 0:
            pattern_error_rates[pattern] = stats['errors']
    
    # High priority: Most problematic patterns
    sorted_patterns = sorted(pattern_error_rates.items(), key=lambda x: x[1], reverse=True)
    if sorted_patterns:
        top_3_patterns = sorted_patterns[:3]
        recommendations['high_priority'].append({
            'issue': 'Most Problematic Patterns',
            'patterns': [p[0] for p in top_3_patterns],
            'error_count': sum(p[1] for p in top_3_patterns),
            'recommendation': f"Focus on improving {', '.join(p[0] for p in top_3_patterns)}. These patterns account for the majority of errors.",
            'actions': [
                f"Create specialized training examples for {top_3_patterns[0][0]}",
                "Add pattern-specific validation rules",
                "Enhance prompt engineering for these patterns"
            ]
        })
    
    # Syntax issues
    total_syntax_errors = sum(mistakes['syntax_patterns'].values())
    if total_syntax_errors > 0:
        recommendations['high_priority'].append({
            'issue': 'Syntax Errors',
            'error_count': total_syntax_errors,
            'breakdown': dict(mistakes['syntax_patterns']),
            'recommendation': "Implement syntax validation and post-processing",
            'actions': [
                "Add parenthesis balancing checks",
                "Implement automatic whitespace normalization",
                "Add empty query detection and retry logic"
            ]
        })
    
    # Temporal logic issues
    total_temporal_errors = sum(mistakes['temporal_logic_errors'].values())
    if total_temporal_errors > 0:
        recommendations['high_priority'].append({
            'issue': 'Temporal Logic Errors',
            'error_count': total_temporal_errors,
            'breakdown': dict(mistakes['temporal_logic_errors']),
            'recommendation': "Strengthen temporal logic understanding",
            'actions': [
                "Clarify TCTL vs CTL differences in prompts",
                "Add examples showing clock variable usage",
                "Include temporal constraint patterns in few-shot examples"
            ]
        })
    
    # Semantic issues
    total_semantic_errors = sum(mistakes['semantic_issues'].values())
    if total_semantic_errors > 0:
        recommendations['medium_priority'].append({
            'issue': 'Semantic Mapping Errors',
            'error_count': total_semantic_errors,
            'breakdown': dict(mistakes['semantic_issues']),
            'recommendation': "Improve natural language to formal logic mapping",
            'actions': [
                "Add keyword-to-operator mapping examples (always→A[], eventually→E<>)",
                "Include negation handling examples",
                "Strengthen semantic verification in validation step"
            ]
        })
    
    # Variable issues
    total_variable_errors = sum(mistakes['variable_issues'].values())
    if total_variable_errors > 0:
        recommendations['medium_priority'].append({
            'issue': 'Variable Handling Errors',
            'error_count': total_variable_errors,
            'breakdown': dict(mistakes['variable_issues']),
            'recommendation': "Improve variable extraction and usage",
            'actions': [
                "Add variable extraction from constraints",
                "Include variable scope validation",
                "Provide examples showing correct variable usage in queries"
            ]
        })
    
    # Model-specific recommendations
    model_error_counts = {model: len(samples) for model, samples in wrong_samples.items()}
    sorted_models = sorted(model_error_counts.items(), key=lambda x: x[1], reverse=True)
    
    for model_key, error_count in sorted_models:
        if error_count > 50:  # High error count
            top_patterns = sorted(
                analysis['model_pattern_errors'][model_key].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            recommendations['model_specific'][model_key].append({
                'issue': f'High error rate ({error_count} errors)',
                'top_failing_patterns': [p[0] for p in top_patterns],
                'recommendation': f"This model struggles with {', '.join(p[0] for p in top_patterns[:2])}",
                'actions': [
                    "Consider using a different model for these patterns",
                    "Add model-specific prompt adjustments",
                    "Increase few-shot examples for problematic patterns"
                ]
            })
    
    # Pattern-specific recommendations
    for pattern, samples in analysis['common_constraint_issues'].items():
        if len(samples) > 30:  # Pattern with many errors
            recommendations['pattern_specific'][pattern].append({
                'issue': f'High failure rate ({len(samples)} errors)',
                'recommendation': f"Pattern '{pattern}' needs significant improvement",
                'actions': [
                    "Review and enhance pattern definition",
                    "Add more diverse training examples",
                    "Consider breaking down into sub-patterns",
                    "Implement pattern-specific validation rules"
                ]
            })
    
    # Scenario hotspots
    top_scenarios = sorted(analysis['scenario_hotspots'].items(), key=lambda x: x[1], reverse=True)[:5]
    if top_scenarios and top_scenarios[0][1] > 10:
        recommendations['medium_priority'].append({
            'issue': 'Scenario Hotspots',
            'scenarios': [s[0] for s in top_scenarios],
            'error_counts': {s[0]: s[1] for s in top_scenarios},
            'recommendation': "Some scenarios are particularly challenging",
            'actions': [
                f"Review {top_scenarios[0][0]} - it has the most errors ({top_scenarios[0][1]})",
                "Analyze what makes these scenarios difficult",
                "Create targeted examples based on these scenarios"
            ]
        })
    
    return recommendations

def create_improvement_plan(recommendations: dict):
    """Create a structured improvement plan"""
    
    plan = {
        'immediate_actions': [],
        'short_term': [],
        'long_term': [],
        'quick_wins': [],
    }
    
    # Immediate: Fix syntax issues (easy wins)
    for rec in recommendations['high_priority']:
        if rec['issue'] == 'Syntax Errors':
            plan['immediate_actions'].append({
                'priority': 'CRITICAL',
                'action': 'Implement syntax validation layer',
                'details': rec['actions'],
                'estimated_impact': 'High - could fix 10-20% of errors'
            })
    
    # Short-term: Improve prompts and examples
    for rec in recommendations['high_priority']:
        if rec['issue'] in ['Temporal Logic Errors', 'Most Problematic Patterns']:
            plan['short_term'].append({
                'priority': 'HIGH',
                'action': f"Address {rec['issue']}",
                'details': rec['actions'],
                'estimated_impact': 'Medium-High - could fix 15-25% of errors'
            })
    
    # Medium-term: Semantic improvements
    for rec in recommendations['medium_priority']:
        plan['short_term'].append({
            'priority': 'MEDIUM',
            'action': f"Improve {rec['issue']}",
            'details': rec['actions'],
            'estimated_impact': 'Medium - could fix 5-15% of errors'
        })
    
    # Long-term: Model selection and pattern refinement
    if recommendations['model_specific']:
        plan['long_term'].append({
            'priority': 'LOW',
            'action': 'Optimize model selection per pattern',
            'details': ['Route patterns to best-performing models', 'Create ensemble approach'],
            'estimated_impact': 'Medium - could improve overall accuracy by 5-10%'
        })
    
    # Quick wins
    plan['quick_wins'] = [
        {
            'action': 'Add parenthesis balancing post-processor',
            'effort': 'Low',
            'impact': 'Medium'
        },
        {
            'action': 'Implement whitespace normalization',
            'effort': 'Low',
            'impact': 'Low-Medium'
        },
        {
            'action': 'Add empty query retry logic',
            'effort': 'Low',
            'impact': 'Low'
        },
        {
            'action': 'Create keyword-to-operator mapping table',
            'effort': 'Medium',
            'impact': 'Medium-High'
        },
    ]
    
    return plan

def save_analysis_report(analysis: dict, mistakes: dict, recommendations: dict, 
                         improvement_plan: dict, output_file: str = "error_analysis_report.json"):
    """Save comprehensive analysis report"""
    
    report = {
        'analysis_date': '2026-01-28',
        'total_wrong_samples': sum(len(samples) for samples in analysis['common_constraint_issues'].values()),
        'error_patterns': {
            'by_pattern': dict(analysis['pattern_errors']),
            'by_scenario': dict(analysis['scenario_hotspots']),
        },
        'common_mistakes': {
            'syntax': dict(mistakes['syntax_patterns']),
            'temporal_logic': dict(mistakes['temporal_logic_errors']),
            'semantic': dict(mistakes['semantic_issues']),
            'variables': dict(mistakes['variable_issues']),
        },
        'recommendations': recommendations,
        'improvement_plan': improvement_plan,
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Analysis report saved to {output_file}")
    return report

def print_analysis_summary(analysis: dict, mistakes: dict, recommendations: dict, improvement_plan: dict):
    """Print comprehensive analysis summary"""
    
    print("\n" + "="*100)
    print("ERROR ANALYSIS & IMPROVEMENT RECOMMENDATIONS")
    print("="*100)
    
    # Overall statistics
    total_errors = sum(v['count'] for v in analysis['pattern_errors'].values())
    print(f"\n📊 OVERALL STATISTICS")
    print("-"*100)
    print(f"Total wrong samples analyzed: {total_errors}")
    print(f"Unique patterns with errors: {len(analysis['pattern_errors'])}")
    print(f"Affected scenarios: {len(analysis['scenario_hotspots'])}")
    
    # Pattern breakdown
    print(f"\n🎯 ERROR BREAKDOWN BY PATTERN")
    print("-"*100)
    sorted_patterns = sorted(analysis['pattern_errors'].items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"{'Pattern':<40} {'Error Count':<15} {'% of Total':<15}")
    print("-"*100)
    for pattern, stats in sorted_patterns[:10]:
        percentage = (stats['count'] / total_errors * 100) if total_errors > 0 else 0
        print(f"{pattern:<40} {stats['count']:<15} {percentage:>6.1f}%")
    
    # Common mistakes
    print(f"\n⚠️  COMMON MISTAKES")
    print("-"*100)
    
    all_mistakes = []
    for category, items in mistakes.items():
        for mistake, count in items.items():
            all_mistakes.append((category, mistake, count))
    
    all_mistakes.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Category':<30} {'Mistake':<40} {'Count':<10}")
    print("-"*100)
    for category, mistake, count in all_mistakes[:15]:
        print(f"{category:<30} {mistake:<40} {count:<10}")
    
    # Top scenarios with errors
    print(f"\n🔥 SCENARIO HOTSPOTS (Top 10)")
    print("-"*100)
    sorted_scenarios = sorted(analysis['scenario_hotspots'].items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"{'Scenario':<30} {'Error Count':<15}")
    print("-"*100)
    for scenario, count in sorted_scenarios:
        print(f"{scenario:<30} {count:<15}")
    
    # High priority recommendations
    print(f"\n🚨 HIGH PRIORITY RECOMMENDATIONS")
    print("-"*100)
    for i, rec in enumerate(recommendations['high_priority'], 1):
        print(f"\n{i}. {rec['issue']}")
        print(f"   Error Count: {rec['error_count']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Actions:")
        for action in rec['actions']:
            print(f"      • {action}")
    
    # Medium priority recommendations
    if recommendations['medium_priority']:
        print(f"\n⚡ MEDIUM PRIORITY RECOMMENDATIONS")
        print("-"*100)
        for i, rec in enumerate(recommendations['medium_priority'], 1):
            print(f"\n{i}. {rec['issue']}")
            if 'error_count' in rec:
                print(f"   Error Count: {rec['error_count']}")
            print(f"   Recommendation: {rec['recommendation']}")
    
    # Improvement plan
    print(f"\n📋 IMPROVEMENT PLAN")
    print("="*100)
    
    print(f"\n🔴 IMMEDIATE ACTIONS (Do First)")
    print("-"*100)
    for i, action in enumerate(improvement_plan['immediate_actions'], 1):
        print(f"\n{i}. {action['action']} [{action['priority']}]")
        print(f"   Estimated Impact: {action['estimated_impact']}")
        print(f"   Steps:")
        for step in action['details']:
            print(f"      • {step}")
    
    print(f"\n🟡 SHORT-TERM ACTIONS (Next Sprint)")
    print("-"*100)
    for i, action in enumerate(improvement_plan['short_term'], 1):
        print(f"\n{i}. {action['action']} [{action['priority']}]")
        print(f"   Estimated Impact: {action['estimated_impact']}")
    
    print(f"\n🟢 QUICK WINS (Easy Improvements)")
    print("-"*100)
    print(f"{'Action':<50} {'Effort':<15} {'Impact':<15}")
    print("-"*100)
    for qw in improvement_plan['quick_wins']:
        print(f"{qw['action']:<50} {qw['effort']:<15} {qw['impact']:<15}")
    
    # Model-specific insights
    if recommendations['model_specific']:
        print(f"\n🤖 MODEL-SPECIFIC INSIGHTS")
        print("-"*100)
        for model, recs in list(recommendations['model_specific'].items())[:5]:
            print(f"\n{model}:")
            for rec in recs:
                print(f"   {rec['issue']}")
                print(f"   Top failing patterns: {', '.join(rec['top_failing_patterns'][:3])}")
    
    print("\n" + "="*100)
    print("💡 KEY TAKEAWAYS")
    print("="*100)
    print("""
1. Focus on syntax validation first - quick wins with high impact
2. Improve temporal logic handling with better examples and prompts
3. Address the top 3 most problematic patterns
4. Consider model routing based on pattern-specific performance
5. Implement incremental improvements and measure impact
    """)
    print("="*100 + "\n")

def main():
    """Main execution"""
    print("Loading wrong samples...")
    wrong_samples, summary = load_wrong_samples()
    
    total_samples = sum(len(samples) for samples in wrong_samples.values())
    print(f"✓ Loaded {total_samples} wrong samples from {len(wrong_samples)} models")
    
    print("\nAnalyzing error patterns...")
    analysis, all_samples = analyze_error_patterns(wrong_samples)
    print(f"✓ Analyzed {len(all_samples)} individual errors")
    
    print("\nIdentifying common mistakes...")
    mistakes = identify_common_mistakes(all_samples)
    print(f"✓ Identified {sum(len(m) for m in mistakes.values())} mistake categories")
    
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(analysis, mistakes, wrong_samples)
    print(f"✓ Generated {len(recommendations['high_priority'])} high-priority recommendations")
    
    print("\nCreating improvement plan...")
    improvement_plan = create_improvement_plan(recommendations)
    print(f"✓ Created action plan with {len(improvement_plan['immediate_actions'])} immediate actions")
    
    print("\nSaving analysis report...")
    report = save_analysis_report(analysis, mistakes, recommendations, improvement_plan)
    
    print_analysis_summary(analysis, mistakes, recommendations, improvement_plan)
    
    print("✅ Analysis complete!")
    print("   - Full report: error_analysis_report.json")
    print("   - Review the recommendations above to improve model performance")

if __name__ == "__main__":
    main()
