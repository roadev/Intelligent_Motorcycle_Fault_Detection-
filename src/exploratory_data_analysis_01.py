"""
Intelligent Motorcycle Fault Detection - Phase 1: Exploratory Data Analysis (EDA)
Authors: Mohammad Kazim, Aryan Gupta, Faiz Maqsood
Bharati Vidyapeeth College of Engineering, Pune

This script performs comprehensive EDA on motorcycle engine audio dataset.
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import random

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MotorcycleAudioEDA:
    def __init__(self, dataset_path, output_dir='./eda/'):
        """
        Initialize EDA class
        
        Args:
            dataset_path (str): Path to the dataset directory
            output_dir (str): Directory to save EDA plots
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.audio_data = []
        self.labels = []
        self.file_paths = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_dataset_info(self):
        """Load dataset information without loading actual audio data"""
        print("📊 Loading dataset information...")
        
        # Assuming files are organized in folders by category or have labels in filename
        # Update this based on your actual dataset structure
        fault_categories = ['engine', 'brake', 'chain', 'silencer', 'normal']
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)
                    
                    # Extract label from filename or folder structure
                    # Adjust this logic based on your dataset organization
                    label = self._extract_label(file_path, fault_categories)
                    self.labels.append(label)
        
        print(f"Found {len(self.file_paths)} audio files")
        return len(self.file_paths)
    
    def _extract_label(self, file_path, categories):
        """Extract label from file path or filename"""
        file_lower = file_path.lower()
        
        for category in categories:
            if category in file_lower:
                return category
        
        # Default to normal if no category found
        return 'normal'
    
    def analyze_class_distribution(self):
        """Analyze and plot class distribution"""
        print("📈 Analyzing class distribution...")
        
        label_counts = Counter(self.labels)
        
        # Create distribution plot
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        categories = list(label_counts.keys())
        counts = list(label_counts.values())
        
        bars = plt.bar(categories, counts, color=sns.color_palette("husl", len(categories)))
        plt.title('Distribution of Fault Categories', fontsize=14, fontweight='bold')
        plt.xlabel('Fault Category', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.title('Fault Category Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print("\n📊 Class Distribution Statistics:")
        for category, count in sorted(label_counts.items()):
            percentage = (count / len(self.labels)) * 100
            print(f"  {category.capitalize()}: {count} samples ({percentage:.1f}%)")
        
        # Check for class imbalance
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 2:
            print(f"\n⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        else:
            print(f"\n✅ Classes are relatively balanced. Ratio: {imbalance_ratio:.2f}")
        
        return label_counts
    
    def visualize_audio_samples(self, samples_per_class=3):
        """Visualize waveforms and mel-spectrograms for random samples"""
        print("🎵 Visualizing audio samples...")
        
        label_counts = Counter(self.labels)
        categories = list(label_counts.keys())
        
        for category in categories:
            print(f"Processing {category} samples...")
            
            # Get files for this category
            category_files = [fp for fp, label in zip(self.file_paths, self.labels) if label == category]
            
            if len(category_files) == 0:
                continue
                
            # Select random samples
            sample_files = random.sample(category_files, min(samples_per_class, len(category_files)))
            
            # Create plots for this category
            fig, axes = plt.subplots(len(sample_files), 2, figsize=(15, 4*len(sample_files)))
            if len(sample_files) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{category.capitalize()} Fault - Audio Analysis', fontsize=16, fontweight='bold')
            
            for idx, file_path in enumerate(sample_files):
                try:
                    # Load audio
                    y, sr = librosa.load(file_path, sr=None, duration=5.0)  # Load first 5 seconds
                    
                    # Plot waveform
                    axes[idx, 0].plot(np.linspace(0, len(y)/sr, len(y)), y)
                    axes[idx, 0].set_title(f'Waveform - {os.path.basename(file_path)}')
                    axes[idx, 0].set_xlabel('Time (s)')
                    axes[idx, 0].set_ylabel('Amplitude')
                    axes[idx, 0].grid(True, alpha=0.3)
                    
                    # Plot mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', 
                                                 sr=sr, ax=axes[idx, 1])
                    axes[idx, 1].set_title(f'Mel-Spectrogram - {os.path.basename(file_path)}')
                    plt.colorbar(img, ax=axes[idx, 1], format='%+2.0f dB')
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{category}_samples_visualization.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def calculate_audio_statistics(self):
        """Calculate basic audio statistics for each file"""
        print("📊 Calculating audio statistics...")
        
        stats_data = []
        
        for file_path, label in zip(self.file_paths, self.labels):
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=None)
                
                # Calculate statistics
                duration = len(y) / sr
                mean_amplitude = np.mean(np.abs(y))
                std_amplitude = np.std(y)
                max_amplitude = np.max(np.abs(y))
                rms_energy = np.sqrt(np.mean(y**2))
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y)[0])
                
                stats_data.append({
                    'filename': os.path.basename(file_path),
                    'label': label,
                    'duration': duration,
                    'sample_rate': sr,
                    'mean_amplitude': mean_amplitude,
                    'std_amplitude': std_amplitude,
                    'max_amplitude': max_amplitude,
                    'rms_energy': rms_energy,
                    'zero_crossing_rate': zero_crossing_rate
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistics
        stats_df.to_csv(os.path.join(self.output_dir, 'audio_statistics.csv'), index=False)
        
        # Plot statistics by category
        self._plot_statistics_by_category(stats_df)
        
        return stats_df
    
    def _plot_statistics_by_category(self, stats_df):
        """Plot statistics grouped by category"""
        numeric_cols = ['duration', 'mean_amplitude', 'std_amplitude', 'rms_energy', 'zero_crossing_rate']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(data=stats_df, x='label', y=col, ax=axes[idx])
            axes[idx].set_title(f'{col.replace("_", " ").title()} by Category', fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Summary statistics table
        axes[5].axis('tight')
        axes[5].axis('off')
        
        summary_stats = stats_df.groupby('label')[numeric_cols].agg(['mean', 'std']).round(4)
        table_data = []
        
        for category in summary_stats.index:
            for col in numeric_cols:
                mean_val = summary_stats.loc[category, (col, 'mean')]
                std_val = summary_stats.loc[category, (col, 'std')]
                table_data.append([category, col, f"{mean_val:.4f}", f"{std_val:.4f}"])
        
        table = axes[5].table(cellText=table_data[:15],  # Show first 15 rows
                             colLabels=['Category', 'Feature', 'Mean', 'Std'],
                             cellLoc='center',
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        axes[5].set_title('Summary Statistics (Sample)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'audio_statistics_by_category.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n📊 Audio Statistics Summary:")
        print(stats_df.groupby('label')[numeric_cols].describe().round(4))
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report"""
        print("📋 Generating EDA Report...")
        
        report = f"""
# Motorcycle Fault Detection - EDA Report

## Dataset Overview
- **Total Files**: {len(self.file_paths)}
- **Categories**: {len(set(self.labels))}
- **Category List**: {', '.join(sorted(set(self.labels)))}

## Class Distribution
{Counter(self.labels)}

## Key Findings
1. **Dataset Size**: {len(self.file_paths)} audio samples
2. **Categories**: {len(set(self.labels))} fault types detected
3. **File Format**: WAV audio files
4. **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Generated
- class_distribution.png: Visual representation of class distribution
- [category]_samples_visualization.png: Sample waveforms and spectrograms for each category
- audio_statistics.csv: Detailed statistics for all audio files
- audio_statistics_by_category.png: Statistical analysis by category

## Recommendations
- Check for class imbalance and apply appropriate techniques if needed
- Consider data augmentation for underrepresented classes
- Verify audio quality and remove noisy samples if necessary
"""
        
        with open(os.path.join(self.output_dir, 'eda_report.md'), 'w') as f:
            f.write(report)
        
        print("✅ EDA Report generated successfully!")
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("🚀 Starting Comprehensive EDA for Motorcycle Fault Detection Dataset")
        print("=" * 70)
        
        # Load dataset info
        total_files = self.load_dataset_info()
        
        if total_files == 0:
            print("❌ No audio files found! Please check the dataset path.")
            return
        
        # Analyze class distribution
        label_counts = self.analyze_class_distribution()
        
        # Visualize samples
        self.visualize_audio_samples()
        
        # Calculate statistics
        stats_df = self.calculate_audio_statistics()
        
        # Generate report
        self.generate_eda_report()
        
        print("\n" + "=" * 70)
        print("✅ EDA Complete! Check the './eda/' folder for all generated plots and reports.")
        
        return {
            'total_files': total_files,
            'label_counts': label_counts,
            'statistics': stats_df
        }

def main():
    """Main function to run EDA"""
    # Update this path to your actual dataset location
    DATASET_PATH = "./data/"  # Change this to your dataset path
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path '{DATASET_PATH}' not found!")
        print("Please update the DATASET_PATH variable with the correct path to your audio files.")
        return
    
    # Initialize and run EDA
    eda = MotorcycleAudioEDA(dataset_path=DATASET_PATH)
    results = eda.run_complete_eda()
    
    return results

if __name__ == "__main__":
    main()