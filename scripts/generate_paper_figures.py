#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_stacked_subagent_breakdown(df_filtered, output_dir):
    fig, ax = plt.subplots(figsize=(14, 7))

    multi_df = df_filtered[df_filtered['agent'] == 'RefinedMultiAgent'].copy()
    pivot = multi_df.pivot(index='task', columns='subagent', values='number_of_steps')
    pivot = pivot.fillna(0)

    subagent_order = ['researcher', 'coder', 'test_generator', 'coordinator']
    pivot = pivot[[col for col in subagent_order if col in pivot.columns]]

    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False)
    pivot = pivot.drop('total', axis=1)

    colors = {
        'researcher': '#3498db',
        'coder': '#2ecc71',
        'test_generator': '#e74c3c',
        'coordinator': '#95a5a6'
    }

    pivot.plot(kind='bar', stacked=True, ax=ax,
              color=[colors[col] for col in pivot.columns],
              width=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Number of Steps', fontweight='bold')
    ax.set_title('Multi-Agent: Steps by Subagent (Stacked)', fontweight='bold', pad=20)
    ax.legend(title='Subagent', loc='upper right', framealpha=0.9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'stacked_subagent_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def plot_grouped_bars(df, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    df['n-weighted speedup'] = pd.to_numeric(df['n-weighted speedup'], errors='coerce')

    tasks = df['Task'].unique()
    agents = df['Agent'].unique()
    x = np.arange(len(tasks))
    width = 0.2

    colors = {
        'SingleAgent': '#2ecc71',
        'MultiAgent': '#3498db',
        'MultiAgentNoResearcher': '#e74c3c',
        'MultiAgentNoResearcherOrTester': '#f39c12'
    }

    for i, agent in enumerate(agents):
        agent_data = df[df['Agent'] == agent].set_index('Task')
        speedups = [agent_data.loc[task, 'n-weighted speedup'] if task in agent_data.index else 0
                   for task in tasks]
        successes = [agent_data.loc[task, 'Success'] if task in agent_data.index else False
                    for task in tasks]

        speedups = [s if not pd.isna(s) else 0 for s in speedups]

        bars = ax.bar(x + i*width, speedups, width,
                     label=agent,
                     color=colors.get(agent, '#999999'),
                     alpha=0.9,
                     edgecolor='black',
                     linewidth=0.5)

        for j, (bar, success) in enumerate(zip(bars, successes)):
            if not success:
                bar.set_alpha(0.3)
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       '✗', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Agent Performance by Task', fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)

    valid_speedups = df['n-weighted speedup'].dropna()
    if len(valid_speedups) > 0:
        ax.set_ylim(0, max(valid_speedups) * 1.15)

    plt.tight_layout()
    output_path = output_dir / 'grouped_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()


def main():
    logs_summary_path = Path('logs_summary.csv')
    agent_comparison_path = Path('data/agent_comparison.csv')

    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    if logs_summary_path.exists():
        df_summary = pd.read_csv(logs_summary_path)

        multi_agent_tasks = df_summary[df_summary['agent'] == 'RefinedMultiAgent']['task'].unique()
        single_agent_tasks = df_summary[df_summary['agent'] == 'NewAgent']['task'].unique()
        common_tasks = sorted(set(multi_agent_tasks) & set(single_agent_tasks))

        df_filtered = df_summary[df_summary['task'].isin(common_tasks)].copy()

        if len(df_filtered) > 0:
            plot_stacked_subagent_breakdown(df_filtered, output_dir)
        else:
            print("No common tasks found for stacked subagent breakdown")
    else:
        print(f"Warning: {logs_summary_path} not found, skipping stacked subagent breakdown")

    if agent_comparison_path.exists():
        df_comparison = pd.read_csv(agent_comparison_path)
        plot_grouped_bars(df_comparison, output_dir)
    else:
        print(f"Warning: {agent_comparison_path} not found, skipping grouped bars")

    print(f"\nFigures saved to {output_dir}/")


if __name__ == '__main__':
    main()
