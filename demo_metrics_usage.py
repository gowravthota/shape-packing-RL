#!/usr/bin/env python3
"""
Demo: How to Load and Use Training Metrics

This script demonstrates how to easily load and work with the training metrics
that are automatically saved during training.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_latest_metrics():
    """Load the most recent metrics data."""
    metrics_dir = Path("metrics")
    
    # Find the most recent JSON file
    json_files = list(metrics_dir.glob("*.json"))
    if not json_files:
        print("No metrics files found! Run training first.")
        return None
    
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading metrics from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data

def demo_basic_analysis():
    """Demonstrate basic analysis of training metrics."""
    print("="*60)
    print("DEMO: Basic Metrics Analysis")
    print("="*60)
    
    # Load data
    data = load_latest_metrics()
    if not data:
        return
    
    # Extract key metrics
    episode_data = data['episode_data']
    summary_metrics = data['summary_metrics']
    
    # Basic statistics
    rewards = [ep['reward'] for ep in episode_data]
    filled_cells = [ep['filled_cells'] for ep in episode_data]
    
    print(f"Training completed with {len(episode_data)} episodes")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average filled cells: {np.mean(filled_cells):.2f}")
    print(f"Best performance: {max(filled_cells)} cells filled")
    
    # Show some sample episodes
    print(f"\nSample episodes:")
    for i in [0, len(episode_data)//2, -1]:
        ep = episode_data[i]
        print(f"  Episode {ep['episode']:3d}: Reward={ep['reward']:6.1f}, Filled={ep['filled_cells']:3d}, Epsilon={ep.get('epsilon', 'N/A')}")

def demo_csv_analysis():
    """Demonstrate loading and analyzing CSV metrics."""
    print("\n" + "="*60)
    print("DEMO: CSV Data Analysis with Pandas")
    print("="*60)
    
    # Find CSV file
    metrics_dir = Path("metrics")
    csv_files = list(metrics_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading CSV from: {latest_csv}")
    
    # Load with pandas
    df = pd.read_csv(latest_csv)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic pandas analysis
    print(f"\nBasic statistics:")
    print(df[['reward', 'filled_cells', 'epsilon']].describe())
    
    # Show correlation between metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        print(f"\nCorrelations:")
        print(df[numeric_cols].corr().round(3))

def demo_simple_plotting():
    """Demonstrate creating simple plots from metrics."""
    print("\n" + "="*60)
    print("DEMO: Simple Plotting")
    print("="*60)
    
    # Load CSV data
    metrics_dir = Path("metrics")
    csv_files = list(metrics_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_csv)
    
    # Create a simple 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Simple Training Metrics Dashboard', fontsize=14)
    
    # Plot 1: Reward over time
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Filled cells distribution
    axes[0, 1].hist(df['filled_cells'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Filled Cells Distribution')
    axes[0, 1].set_xlabel('Filled Cells')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    if 'epsilon' in df.columns:
        axes[1, 0].plot(df['episode'], df['epsilon'].dropna(), color='red')
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training loss
    if 'loss' in df.columns:
        loss_data = df['loss'].dropna()
        if len(loss_data) > 0:
            axes[1, 1].plot(range(len(loss_data)), loss_data, color='purple', alpha=0.7)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "metrics/simple_dashboard.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Simple dashboard saved to: {output_path}")
    plt.show()

def demo_custom_analysis():
    """Demonstrate custom analysis examples."""
    print("\n" + "="*60)
    print("DEMO: Custom Analysis Examples")
    print("="*60)
    
    data = load_latest_metrics()
    if not data:
        return
    
    episode_data = data['episode_data']
    
    # Example 1: Learning efficiency (episodes to reach certain performance)
    target_cells = 95  # Target filled cells
    for i, ep in enumerate(episode_data):
        if ep['filled_cells'] >= target_cells:
            print(f"Reached {target_cells}+ filled cells in episode {ep['episode']}")
            break
    else:
        print(f"Never reached {target_cells}+ filled cells")
    
    # Example 2: Episode-by-episode improvement tracking
    rewards = [ep['reward'] for ep in episode_data]
    window_size = 50
    if len(rewards) >= window_size:
        early_avg = np.mean(rewards[:window_size])
        late_avg = np.mean(rewards[-window_size:])
        improvement = late_avg - early_avg
        print(f"Improvement over training: {improvement:.2f} reward points")
        print(f"  (comparing first {window_size} vs last {window_size} episodes)")
    
    # Example 3: Find best and worst episodes
    best_ep = max(episode_data, key=lambda x: x['filled_cells'])
    worst_ep = min(episode_data, key=lambda x: x['filled_cells'])
    print(f"\nBest episode: #{best_ep['episode']} (filled {best_ep['filled_cells']} cells)")
    print(f"Worst episode: #{worst_ep['episode']} (filled {worst_ep['filled_cells']} cells)")

def main():
    """Run all demos."""
    print("Space RL Metrics Usage Demo")
    print("This demo shows how to load and analyze training metrics.")
    
    try:
        demo_basic_analysis()
        demo_csv_analysis()
        demo_simple_plotting()
        demo_custom_analysis()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("Key takeaways:")
        print("• Metrics are automatically saved as JSON and CSV during training")
        print("• JSON contains detailed episode data and metadata")
        print("• CSV is perfect for pandas analysis and custom plotting")
        print("• Use analyze_metrics.py for comprehensive analysis")
        print("• Create custom analysis scripts for specific research questions")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure you've run training first to generate metrics!")

if __name__ == "__main__":
    main() 