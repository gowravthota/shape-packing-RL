#!/usr/bin/env python3
"""
Metrics Analysis and Visualization Tool for Space RL

This script loads saved training metrics and generates comprehensive visualizations
and analysis reports.

Usage:
    python analyze_metrics.py [--json_file path/to/metrics.json] [--csv_file path/to/metrics.csv]
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Optional

plt.style.use('default')
sns.set_palette("husl")

class MetricsAnalyzer:
    """Analyze and visualize training metrics from saved files."""
    
    def __init__(self, json_file: str = None, csv_file: str = None):
        self.json_file = json_file
        self.csv_file = csv_file
        self.metrics_data = None
        self.episode_data = None
        
        if json_file and os.path.exists(json_file):
            self.load_json_metrics(json_file)
        elif csv_file and os.path.exists(csv_file):
            self.load_csv_metrics(csv_file)
        else:
            # Try to find the most recent metrics file
            self.find_latest_metrics()
    
    def find_latest_metrics(self):
        """Find the most recent metrics files in the metrics directory."""
        metrics_dir = Path("metrics")
        if not metrics_dir.exists():
            print("No metrics directory found!")
            return
        
        json_files = list(metrics_dir.glob("*.json"))
        if json_files:
            # Get the most recent JSON file
            latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
            self.json_file = str(latest_json)
            self.load_json_metrics(self.json_file)
            print(f"Loaded metrics from: {self.json_file}")
        else:
            csv_files = list(metrics_dir.glob("*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
                self.csv_file = str(latest_csv)
                self.load_csv_metrics(self.csv_file)
                print(f"Loaded metrics from: {self.csv_file}")
    
    def load_json_metrics(self, json_file: str):
        """Load metrics from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.metrics_data = data.get('summary_metrics', {})
        self.episode_data = data.get('episode_data', [])
        self.session_info = data.get('session_info', {})
    
    def load_csv_metrics(self, csv_file: str):
        """Load metrics from CSV file."""
        df = pd.read_csv(csv_file)
        
        # Convert to our internal format
        self.episode_data = df.to_dict('records')
        self.metrics_data = {
            'episode_rewards': df['reward'].tolist(),
            'average_reward': df['avg_reward'].dropna().tolist(),
            'filled_cells': df['filled_cells'].tolist(),
            'success_rate': df['success_rate'].dropna().tolist(),
            'epsilon_values': df['epsilon'].dropna().tolist(),
            'loss_values': df['loss'].dropna().tolist(),
        }
    
    def print_summary(self):
        """Print training summary statistics."""
        if not self.episode_data:
            print("No episode data available!")
            return
        
        rewards = [ep.get('reward', 0) for ep in self.episode_data]
        filled_cells = [ep.get('filled_cells', 0) for ep in self.episode_data]
        
        print("\n" + "="*60)
        print("TRAINING METRICS SUMMARY")
        print("="*60)
        
        if hasattr(self, 'session_info') and self.session_info:
            print(f"Session ID: {self.session_info.get('session_id', 'Unknown')}")
            print(f"Start Time: {self.session_info.get('start_time', 'Unknown')}")
        
        print(f"Total Episodes: {len(self.episode_data)}")
        print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Best Reward: {np.max(rewards):.2f}")
        print(f"Worst Reward: {np.min(rewards):.2f}")
        
        print(f"\nAverage Filled Cells: {np.mean(filled_cells):.2f} ± {np.std(filled_cells):.2f}")
        print(f"Best Filled Cells: {np.max(filled_cells)}")
        print(f"Worst Filled Cells: {np.min(filled_cells)}")
        
        # Success rate (positive rewards)
        successes = [1 for r in rewards if r > 0]
        success_rate = len(successes) / len(rewards) if rewards else 0
        print(f"Overall Success Rate: {success_rate:.2%}")
        
        # Learning progress (compare first 25% vs last 25%)
        n = len(rewards)
        if n >= 100:
            early_rewards = rewards[:n//4]
            late_rewards = rewards[-n//4:]
            print(f"\nLearning Progress:")
            print(f"  Early episodes (first 25%): {np.mean(early_rewards):.2f}")
            print(f"  Late episodes (last 25%): {np.mean(late_rewards):.2f}")
            print(f"  Improvement: {np.mean(late_rewards) - np.mean(early_rewards):.2f}")
    
    def plot_comprehensive_analysis(self, save_path: str = None):
        """Create comprehensive analysis plots."""
        if not self.metrics_data:
            print("No metrics data available for plotting!")
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        episodes = list(range(1, len(self.metrics_data['episode_rewards']) + 1))
        
        # 1. Main reward plot (spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.plot(episodes, self.metrics_data['episode_rewards'], alpha=0.6, linewidth=1, 
                label='Episode Rewards', color='skyblue')
        if self.metrics_data.get('average_reward'):
            ax1.plot(episodes[:len(self.metrics_data['average_reward'])], 
                    self.metrics_data['average_reward'], 
                    linewidth=3, label='Moving Average (100 ep)', color='navy')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Filled cells distribution
        ax2 = fig.add_subplot(gs[0, 2])
        filled_cells = self.metrics_data['filled_cells']
        ax2.hist(filled_cells, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Filled Cells')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of\nFilled Cells')
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate over time
        ax3 = fig.add_subplot(gs[0, 3])
        if self.metrics_data.get('success_rate'):
            ax3.plot(episodes[:len(self.metrics_data['success_rate'])], 
                    self.metrics_data['success_rate'], color='orange', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Success Rate\n(Moving Average)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Epsilon decay
        ax4 = fig.add_subplot(gs[1, 2])
        if self.metrics_data.get('epsilon_values'):
            ax4.plot(episodes[:len(self.metrics_data['epsilon_values'])], 
                    self.metrics_data['epsilon_values'], color='red', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Epsilon')
            ax4.set_title('Exploration Rate\n(Epsilon Decay)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Training loss
        ax5 = fig.add_subplot(gs[1, 3])
        if self.metrics_data.get('loss_values'):
            # Smooth the loss for better visualization
            loss_values = self.metrics_data['loss_values']
            if len(loss_values) > 10:
                smoothed_loss = pd.Series(loss_values).rolling(window=10, min_periods=1).mean()
                ax5.plot(range(len(smoothed_loss)), smoothed_loss, color='purple', linewidth=2)
            else:
                ax5.plot(range(len(loss_values)), loss_values, color='purple', linewidth=2)
            ax5.set_xlabel('Training Step')
            ax5.set_ylabel('Loss')
            ax5.set_title('Training Loss\n(Smoothed)')
            ax5.grid(True, alpha=0.3)
        
        # 6. Reward distribution by episode ranges
        ax6 = fig.add_subplot(gs[2, 0])
        rewards = self.metrics_data['episode_rewards']
        n = len(rewards)
        if n >= 100:
            early = rewards[:n//3]
            middle = rewards[n//3:2*n//3]
            late = rewards[2*n//3:]
            
            ax6.boxplot([early, middle, late], labels=['Early', 'Middle', 'Late'])
            ax6.set_ylabel('Reward')
            ax6.set_title('Reward Distribution\nby Training Phase')
            ax6.grid(True, alpha=0.3)
        
        # 7. Filled cells over time
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(episodes, filled_cells, alpha=0.6, color='green')
        # Add trend line
        z = np.polyfit(episodes, filled_cells, 1)
        p = np.poly1d(z)
        ax7.plot(episodes, p(episodes), "--", alpha=0.8, color='darkgreen', linewidth=2)
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Filled Cells')
        ax7.set_title('Filled Cells Over Time\n(with trend)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Performance correlation heatmap
        ax8 = fig.add_subplot(gs[2, 2:])
        if len(self.episode_data) > 10:
            # Create correlation matrix
            df_corr = pd.DataFrame(self.episode_data)
            numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
            corr_matrix = df_corr[numeric_cols].corr()
            
            im = ax8.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax8.set_xticks(range(len(numeric_cols)))
            ax8.set_yticks(range(len(numeric_cols)))
            ax8.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax8.set_yticklabels(numeric_cols)
            ax8.set_title('Metrics Correlation Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
            cbar.set_label('Correlation Coefficient')
        
        # 9. Learning curve with confidence intervals
        ax9 = fig.add_subplot(gs[3, :])
        if len(rewards) > 50:
            window_size = max(10, len(rewards) // 20)
            rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            rolling_std = pd.Series(rewards).rolling(window=window_size, min_periods=1).std()
            
            ax9.plot(episodes, rewards, alpha=0.3, color='lightblue', linewidth=1)
            ax9.plot(episodes, rolling_mean, color='darkblue', linewidth=3, label=f'Rolling Mean ({window_size} ep)')
            ax9.fill_between(episodes, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.2, color='blue', label='±1 Std Dev')
            
            ax9.set_xlabel('Episode')
            ax9.set_ylabel('Reward')
            ax9.set_title('Learning Curve with Confidence Intervals', fontsize=14, fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Training Metrics Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plots saved to: {save_path}")
        
        plt.show()
        return fig
    
    def export_summary_report(self, output_file: str = None):
        """Export a detailed summary report."""
        if not output_file:
            output_file = "metrics/training_analysis_report.txt"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("SPACE RL TRAINING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if hasattr(self, 'session_info') and self.session_info:
                f.write(f"Session ID: {self.session_info.get('session_id', 'Unknown')}\n")
                f.write(f"Training Start: {self.session_info.get('start_time', 'Unknown')}\n\n")
            
            if self.episode_data:
                rewards = [ep.get('reward', 0) for ep in self.episode_data]
                filled_cells = [ep.get('filled_cells', 0) for ep in self.episode_data]
                
                f.write(f"BASIC STATISTICS\n")
                f.write(f"Total Episodes: {len(self.episode_data)}\n")
                f.write(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
                f.write(f"Reward Range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]\n")
                f.write(f"Average Filled Cells: {np.mean(filled_cells):.2f}\n")
                f.write(f"Best Performance: {np.max(filled_cells)} cells filled\n\n")
                
                # Percentile analysis
                f.write(f"PERFORMANCE PERCENTILES\n")
                for p in [25, 50, 75, 90, 95]:
                    f.write(f"{p}th percentile reward: {np.percentile(rewards, p):.2f}\n")
                f.write("\n")
                
                # Learning phases analysis
                n = len(rewards)
                if n >= 100:
                    phase_size = n // 4
                    phases = ['Early', 'Mid-Early', 'Mid-Late', 'Late']
                    f.write(f"LEARNING PHASE ANALYSIS\n")
                    for i, phase in enumerate(phases):
                        start_idx = i * phase_size
                        end_idx = (i + 1) * phase_size if i < 3 else n
                        phase_rewards = rewards[start_idx:end_idx]
                        f.write(f"{phase} ({start_idx+1}-{end_idx}): {np.mean(phase_rewards):.2f} ± {np.std(phase_rewards):.2f}\n")
                    f.write("\n")
        
        print(f"Analysis report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Space RL training metrics')
    parser.add_argument('--json_file', type=str, help='Path to JSON metrics file')
    parser.add_argument('--csv_file', type=str, help='Path to CSV metrics file')
    parser.add_argument('--output_dir', type=str, default='metrics', help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MetricsAnalyzer(json_file=args.json_file, csv_file=args.csv_file)
    
    if not analyzer.metrics_data and not analyzer.episode_data:
        print("No metrics data found! Please run training first or specify a valid metrics file.")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, "comprehensive_analysis.png")
    analyzer.plot_comprehensive_analysis(save_path=plot_path)
    
    # Export report
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    analyzer.export_summary_report(report_path)
    
    print(f"\nAnalysis complete! Check '{args.output_dir}' for all outputs.")

if __name__ == "__main__":
    main() 