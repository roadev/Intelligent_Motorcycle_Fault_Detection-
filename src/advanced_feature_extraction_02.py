"""
Intelligent Motorcycle Fault Detection - Phase 1: Advanced Feature Extraction
Authors: Mohammad Kazim, Aryan Gupta, Faiz Maqsood
Bharati Vidyapeeth College of Engineering, Pune

This script implements comprehensive feature extraction pipeline for motorcycle engine audio analysis.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AdvancedFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=20, n_mels=128, hop_length=512):
        """
        Initialize the feature extractor
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            n_mfcc (int): Number of MFCC coefficients
            n_mels (int): Number of mel bands
            hop_length (int): Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.feature_names = []
        
    def extract_mfcc_features(self, y, sr):
        """Extract MFCC features with statistical measures"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        
        # Statistical measures
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_var = np.var(mfccs, axis=1)
        mfcc_skew = stats.skew(mfccs, axis=1)
        mfcc_kurtosis = stats.kurtosis(mfccs, axis=1)
        
        # Combine features
        mfcc_features = np.concatenate([mfcc_mean, mfcc_var, mfcc_skew, mfcc_kurtosis])
        
        # Feature names
        feature_names = []
        for stat in ['mean', 'var', 'skew', 'kurtosis']:
            for i in range(self.n_mfcc):
                feature_names.append(f'mfcc_{i+1}_{stat}')
        
        return mfcc_features, feature_names
    
    def extract_mel_spectrogram_features(self, y, sr):
        """Extract Mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Statistical measures
        mel_mean = np.mean(mel_spec_db, axis=1)
        mel_var = np.var(mel_spec_db, axis=1)
        
        # Additional statistics
        mel_max = np.max(mel_spec_db, axis=1)
        mel_min = np.min(mel_spec_db, axis=1)
        
        # Combine features
        mel_features = np.concatenate([mel_mean, mel_var, mel_max, mel_min])
        
        # Feature names
        feature_names = []
        for stat in ['mean', 'var', 'max', 'min']:
            for i in range(self.n_mels):
                feature_names.append(f'mel_{i+1}_{stat}')
        
        return mel_features, feature_names
    
    def extract_spectral_features(self, y, sr):
        """Extract comprehensive spectral features"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        
        # Calculate statistics for each feature
        features = []
        feature_names = []
        
        # Process each spectral feature
        spectral_data = {
            'centroid': spectral_centroids,
            'bandwidth': spectral_bandwidth,
            'rolloff': spectral_rolloff,
            'flatness': spectral_flatness
        }
        
        for name, data in spectral_data.items():
            # Statistical measures
            features.extend([
                np.mean(data),
                np.var(data),
                np.std(data),
                np.max(data),
                np.min(data),
                stats.skew(data),
                stats.kurtosis(data)
            ])
            
            for stat in ['mean', 'var', 'std', 'max', 'min', 'skew', 'kurtosis']:
                feature_names.append(f'spectral_{name}_{stat}')
        
        # Process spectral contrast (multiple bands)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        contrast_var = np.var(spectral_contrast, axis=1)
        
        features.extend(contrast_mean)
        features.extend(contrast_var)
        
        for i in range(len(contrast_mean)):
            feature_names.append(f'spectral_contrast_band_{i+1}_mean')
        for i in range(len(contrast_var)):
            feature_names.append(f'spectral_contrast_band_{i+1}_var')
        
        return np.array(features), feature_names
    
    def extract_chroma_features(self, y, sr):
        """Extract chromagram features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        
        # Statistical measures
        chroma_mean = np.mean(chroma, axis=1)
        chroma_var = np.var(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Combine features
        chroma_features = np.concatenate([chroma_mean, chroma_var, chroma_std])
        
        # Feature names
        feature_names = []
        for stat in ['mean', 'var', 'std']:
            for i in range(12):  # 12 chroma bins
                feature_names.append(f'chroma_{i+1}_{stat}')
        
        return chroma_features, feature_names
    
    def extract_temporal_features(self, y, sr):
        """Extract temporal domain features"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        # Handle librosa 0.10+ where tempo is returned as a numpy array
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
        
        # Statistical measures for ZCR and RMS
        features = [
            # ZCR statistics
            np.mean(zcr),
            np.var(zcr),
            np.std(zcr),
            np.max(zcr),
            np.min(zcr),
            
            # RMS statistics
            np.mean(rms),
            np.var(rms),
            np.std(rms),
            np.max(rms),
            np.min(rms),
            
            # Tempo
            tempo,
            
            # Beat-related features
            len(beats),  # Number of beats
            np.mean(np.diff(beats)) if len(beats) > 1 else 0,  # Average beat interval
        ]
        
        feature_names = [
            'zcr_mean', 'zcr_var', 'zcr_std', 'zcr_max', 'zcr_min',
            'rms_mean', 'rms_var', 'rms_std', 'rms_max', 'rms_min',
            'tempo', 'beat_count', 'avg_beat_interval'
        ]
        
        return np.array(features), feature_names
    
    def extract_tonnetz_features(self, y, sr):
        """Extract tonal centroid features (Tonnetz)"""
        # First compute chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Then compute tonnetz
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        
        # Statistical measures
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_var = np.var(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        
        # Combine features
        tonnetz_features = np.concatenate([tonnetz_mean, tonnetz_var, tonnetz_std])
        
        # Feature names (6 tonnetz dimensions)
        feature_names = []
        for stat in ['mean', 'var', 'std']:
            for i in range(6):
                feature_names.append(f'tonnetz_{i+1}_{stat}')
        
        return tonnetz_features, feature_names
    
    def extract_additional_features(self, y, sr):
        """Extract additional advanced features"""
        features = []
        feature_names = []
        
        # Signal energy and power
        signal_energy = np.sum(y**2)
        signal_power = signal_energy / len(y)
        
        # Signal entropy (spectral)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        power_spec = magnitude**2
        power_spec_norm = power_spec / np.sum(power_spec, axis=0, keepdims=True)
        spectral_entropy = -np.sum(power_spec_norm * np.log2(power_spec_norm + 1e-10), axis=0)
        
        # Harmonics and percussives
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-10)
        percussive_ratio = np.sum(y_percussive**2) / (np.sum(y**2) + 1e-10)
        
        # Add features
        features.extend([
            signal_energy,
            signal_power,
            np.mean(spectral_entropy),
            np.var(spectral_entropy),
            harmonic_ratio,
            percussive_ratio
        ])
        
        feature_names.extend([
            'signal_energy',
            'signal_power',
            'spectral_entropy_mean',
            'spectral_entropy_var',
            'harmonic_ratio',
            'percussive_ratio'
        ])
        
        return np.array(features), feature_names
    
    def extract_all_features(self, file_path, max_duration=30):
        """
        Extract all features from a single audio file
        
        Args:
            file_path (str): Path to the audio file
            max_duration (float): Maximum duration to process (seconds)
            
        Returns:
            tuple: (features_array, feature_names_list)
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=max_duration)
            
            # If audio is too short, pad it
            if len(y) < self.sample_rate:
                y = np.pad(y, (0, self.sample_rate - len(y)), 'constant')
            
            all_features = []
            all_feature_names = []
            
            # Extract each type of feature
            feature_extractors = [
                self.extract_mfcc_features,
                self.extract_spectral_features,
                self.extract_chroma_features,
                self.extract_temporal_features,
                self.extract_tonnetz_features,
                self.extract_additional_features
            ]
            
            for extractor in feature_extractors:
                features, names = extractor(y, sr)
                all_features.extend(features)
                all_feature_names.extend(names)
            
            # Mel-spectrogram features (separate due to size)
            mel_features, mel_names = self.extract_mel_spectrogram_features(y, sr)
            
            # Combine with reduced mel features (use only mean and var)
            mel_mean = mel_features[:self.n_mels]
            mel_var = mel_features[self.n_mels:2*self.n_mels]
            
            all_features.extend(mel_mean)
            all_features.extend(mel_var)
            
            mel_feature_names = []
            for stat in ['mean', 'var']:
                for i in range(self.n_mels):
                    mel_feature_names.append(f'mel_{i+1}_{stat}')
            
            all_feature_names.extend(mel_feature_names)
            
            return np.array(all_features), all_feature_names
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, None
    
    def process_dataset(self, dataset_path, output_path='./data/', label_extractor=None):
        """
        Process entire dataset and extract features
        
        Args:
            dataset_path (str): Path to dataset directory
            output_path (str): Path to save processed features
            label_extractor (function): Function to extract labels from file paths
        """
        print("🚀 Starting Advanced Feature Extraction...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print("❌ No audio files found!")
            return None
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process files
        all_features = []
        all_labels = []
        processed_files = []
        
        for file_path in tqdm(audio_files, desc="Extracting features"):
            features, feature_names = self.extract_all_features(file_path)
            
            if features is not None:
                all_features.append(features)
                processed_files.append(os.path.basename(file_path))
                
                # Extract label
                if label_extractor:
                    label = label_extractor(file_path)
                else:
                    label = self._default_label_extractor(file_path)
                
                all_labels.append(label)
        
        if not all_features:
            print("❌ No features extracted!")
            return None
        
        # Convert to arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # Store feature names
        self.feature_names = feature_names
        
        print(f"✅ Extracted {features_array.shape[1]} features from {features_array.shape[0]} files")
        
        # Create DataFrame
        feature_df = pd.DataFrame(features_array, columns=feature_names)
        feature_df['filename'] = processed_files
        feature_df['label'] = labels_array
        
        # Save features
        feature_df.to_csv(os.path.join(output_path, 'extracted_features.csv'), index=False)
        
        # Save as numpy arrays
        np.save(os.path.join(output_path, 'features.npy'), features_array)
        np.save(os.path.join(output_path, 'labels.npy'), labels_array)
        
        # Save feature names
        with open(os.path.join(output_path, 'feature_names.txt'), 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # Save the extractor for later use
        joblib.dump(self, os.path.join(output_path, 'feature_extractor.pkl'))
        
        # Generate feature analysis
        self._analyze_features(feature_df, output_path)
        
        print(f"💾 Features saved to {output_path}")
        
        return feature_df
    
    def _default_label_extractor(self, file_path):
        """Default label extraction from file path"""
        file_lower = file_path.lower()
        categories = ['engine', 'brake', 'chain', 'silencer', 'normal']
        
        for category in categories:
            if category in file_lower:
                return category
        
        return 'normal'
    
    def _analyze_features(self, feature_df, output_path):
        """Analyze and visualize extracted features"""
        print("📊 Analyzing extracted features...")
        
        # Feature correlation analysis
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = feature_df[numeric_features].corr()
        
        # Plot correlation heatmap (for subset of features)
        plt.figure(figsize=(15, 12))
        
        # Select a subset of features for visualization
        important_features = [col for col in numeric_features if any(keyword in col for keyword in 
                            ['mfcc', 'spectral_centroid', 'spectral_bandwidth', 'chroma', 'zcr', 'rms'])][:50]
        
        if important_features:
            subset_corr = correlation_matrix.loc[important_features, important_features]
            sns.heatmap(subset_corr, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix (Subset)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Feature distribution by class
        self._plot_feature_distributions(feature_df, output_path)
        
        # Feature importance analysis (simple variance-based)
        self._analyze_feature_importance(feature_df, output_path)
    
    def _plot_feature_distributions(self, feature_df, output_path):
        """Plot feature distributions by class"""
        # Select key features for visualization
        key_features = [col for col in feature_df.columns if any(keyword in col for keyword in 
                       ['mfcc_1_mean', 'spectral_centroid_mean', 'zcr_mean', 'rms_mean', 'chroma_1_mean'])]
        
        if not key_features:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(key_features[:6]):
            if idx < len(axes):
                sns.boxplot(data=feature_df, x='label', y=feature, ax=axes[idx])
                axes[idx].set_title(f'{feature}', fontweight='bold')
                axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(key_features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'feature_distributions_by_class.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_feature_importance(self, feature_df, output_path):
        """Simple feature importance analysis based on variance"""
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns
        
        # Calculate variance for each feature
        feature_variance = feature_df[numeric_features].var().sort_values(ascending=False)
        
        # Plot top 20 features by variance
        plt.figure(figsize=(12, 8))
        top_features = feature_variance.head(20)
        
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Variance')
        plt.title('Top 20 Features by Variance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, 'feature_importance_variance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance
        feature_variance.to_csv(os.path.join(output_path, 'feature_variance.csv'))

def main():
    """Main function to run feature extraction"""
    # Configuration
    DATASET_PATH = "./data/"  # Update with your dataset path
    OUTPUT_PATH = "./data/"
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path '{DATASET_PATH}' not found!")
        print("Please update the DATASET_PATH variable with the correct path to your audio files.")
        return
    
    # Initialize feature extractor
    extractor = AdvancedFeatureExtractor(
        sample_rate=22050,
        n_mfcc=20,
        n_mels=128,
        hop_length=512
    )
    
    # Process dataset
    feature_df = extractor.process_dataset(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH
    )
    
    if feature_df is not None:
        print("\n✅ Feature extraction completed successfully!")
        print(f"📊 Feature matrix shape: {feature_df.shape}")
        print(f"🎯 Features extracted: {len(extractor.feature_names)}")
        print(f"📁 Files processed: {len(feature_df)}")
        
        # Display feature summary
        print("\n📋 Feature Categories Summary:")
        feature_categories = {}
        for name in extractor.feature_names:
            category = name.split('_')[0]
            feature_categories[category] = feature_categories.get(category, 0) + 1
        
        for category, count in sorted(feature_categories.items()):
            print(f"  {category}: {count} features")
    
    return feature_df

if __name__ == "__main__":
    main()