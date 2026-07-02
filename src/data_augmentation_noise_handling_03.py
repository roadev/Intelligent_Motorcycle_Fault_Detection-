"""
Intelligent Motorcycle Fault Detection - Phase 1: Data Augmentation & Noise Handling
Authors: Mohammad Kazim, Aryan Gupta, Faiz Maqsood
Bharati Vidyapeeth College of Engineering, Pune

This script implements comprehensive data augmentation techniques and noise reduction
for motorcycle engine audio analysis.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.effects
import soundfile as sf
from scipy.signal import convolve
from scipy.io.wavfile import write
import random
from tqdm import tqdm
import warnings

# Try to import noisereduce (optional)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  noisereduce not available. Install with: pip install noisereduce")

warnings.filterwarnings('ignore')

class DataAugmentationPipeline:
    def __init__(self, sample_rate=22050, augmentation_factor=4):
        """
        Initialize data augmentation pipeline
        
        Args:
            sample_rate (int): Target sample rate
            augmentation_factor (int): Number of augmented versions per original file
        """
        self.sample_rate = sample_rate
        self.augmentation_factor = augmentation_factor
        
    def add_gaussian_noise(self, audio, snr_db_range=(10, 30)):
        """
        Add Gaussian noise to audio signal
        
        Args:
            audio (np.array): Input audio signal
            snr_db_range (tuple): Range of SNR values in dB
            
        Returns:
            np.array: Audio with added noise
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Random SNR in the specified range
        snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise power
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        # Add noise to signal
        noisy_audio = audio + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val
        
        return noisy_audio
    
    def time_stretch(self, audio, stretch_factor_range=(0.8, 1.2)):
        """
        Apply time stretching (speed change without pitch change)
        
        Args:
            audio (np.array): Input audio signal
            stretch_factor_range (tuple): Range of stretch factors
            
        Returns:
            np.array: Time-stretched audio
        """
        stretch_factor = random.uniform(stretch_factor_range[0], stretch_factor_range[1])
        
        try:
            stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            return stretched_audio
        except Exception as e:
            print(f"Time stretch failed: {e}")
            return audio
    
    def pitch_shift(self, audio, sr, pitch_shift_range=(-2, 2)):
        """
        Apply pitch shifting
        
        Args:
            audio (np.array): Input audio signal
            sr (int): Sample rate
            pitch_shift_range (tuple): Range of pitch shift in semitones
            
        Returns:
            np.array: Pitch-shifted audio
        """
        n_steps = random.uniform(pitch_shift_range[0], pitch_shift_range[1])
        
        try:
            pitched_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            return pitched_audio
        except Exception as e:
            print(f"Pitch shift failed: {e}")
            return audio
    
    def add_room_impulse_response(self, audio, sr, room_types=['small_room', 'large_hall', 'garage']):
        """
        Simulate room impulse response (reverb effect)
        
        Args:
            audio (np.array): Input audio signal
            sr (int): Sample rate
            room_types (list): Types of room responses to simulate
            
        Returns:
            np.array: Audio with simulated room response
        """
        room_type = random.choice(room_types)
        
        # Generate synthetic impulse response based on room type
        if room_type == 'small_room':
            # Small room: short reverb
            impulse_length = int(0.5 * sr)  # 0.5 seconds
            decay_rate = 0.3
        elif room_type == 'large_hall':
            # Large hall: long reverb
            impulse_length = int(1.5 * sr)  # 1.5 seconds
            decay_rate = 0.1
        else:  # garage
            # Garage: medium reverb with flutter
            impulse_length = int(0.8 * sr)  # 0.8 seconds
            decay_rate = 0.2
        
        # Create exponentially decaying noise as impulse response
        t = np.arange(impulse_length)
        envelope = np.exp(-decay_rate * t / sr)
        
        # Add some randomness to make it more realistic
        impulse = envelope * np.random.normal(0, 0.1, impulse_length)
        impulse[0] = 1.0  # Direct signal
        
        # Convolve with impulse response
        try:
            convolved_audio = convolve(audio, impulse, mode='same')
            
            # Normalize
            max_val = np.max(np.abs(convolved_audio))
            if max_val > 1.0:
                convolved_audio = convolved_audio / max_val
            
            return convolved_audio
        except Exception as e:
            print(f"Room impulse response failed: {e}")
            return audio
    
    def apply_dynamic_range_compression(self, audio, threshold=0.5, ratio=4):
        """
        Apply dynamic range compression
        
        Args:
            audio (np.array): Input audio signal
            threshold (float): Compression threshold
            ratio (float): Compression ratio
            
        Returns:
            np.array: Compressed audio
        """
        # Simple compression algorithm
        compressed = np.copy(audio)
        
        # Find samples above threshold
        above_threshold = np.abs(audio) > threshold
        
        # Apply compression to samples above threshold
        compressed[above_threshold] = (
            threshold + (audio[above_threshold] - threshold) / ratio
        )
        
        return compressed
    
    def reduce_noise(self, audio, sr):
        """
        Reduce background noise using spectral subtraction or noisereduce
        
        Args:
            audio (np.array): Input audio signal
            sr (int): Sample rate
            
        Returns:
            np.array: Noise-reduced audio
        """
        if NOISEREDUCE_AVAILABLE:
            try:
                # Use noisereduce library
                reduced_noise = nr.reduce_noise(y=audio, sr=sr)
                return reduced_noise
            except Exception as e:
                print(f"Noise reduction failed: {e}")
                return self._spectral_subtraction_noise_reduction(audio, sr)
        else:
            return self._spectral_subtraction_noise_reduction(audio, sr)
    
    def _spectral_subtraction_noise_reduction(self, audio, sr):
        """
        Simple spectral subtraction for noise reduction
        
        Args:
            audio (np.array): Input audio signal
            sr (int): Sample rate
            
        Returns:
            np.array: Noise-reduced audio
        """
        try:
            # Estimate noise from first 0.5 seconds (assuming it's mostly noise)
            noise_sample_length = int(0.5 * sr)
            if len(audio) > noise_sample_length:
                noise_sample = audio[:noise_sample_length]
            else:
                noise_sample = audio[:len(audio)//4]  # Use first quarter if too short
            
            # Compute STFT
            stft_audio = librosa.stft(audio)
            stft_noise = librosa.stft(noise_sample)
            
            # Estimate noise spectrum (mean magnitude)
            noise_spectrum = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
            
            # Spectral subtraction
            magnitude = np.abs(stft_audio)
            phase = np.angle(stft_audio)
            
            # Subtract noise spectrum
            clean_magnitude = magnitude - 0.5 * noise_spectrum
            
            # Ensure non-negative values
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            print(f"Spectral subtraction failed: {e}")
            return audio
    
    def augment_single_file(self, audio, sr, augmentation_techniques=None):
        """
        Apply random augmentation techniques to a single audio file
        
        Args:
            audio (np.array): Input audio signal
            sr (int): Sample rate
            augmentation_techniques (list): List of techniques to apply
            
        Returns:
            list: List of augmented audio signals
        """
        if augmentation_techniques is None:
            augmentation_techniques = [
                'gaussian_noise',
                'time_stretch',
                'pitch_shift',
                'room_impulse',
                'compression'
            ]
        
        augmented_audios = []
        
        for _ in range(self.augmentation_factor):
            current_audio = np.copy(audio)
            
            # Randomly select and apply techniques
            num_techniques = random.randint(1, min(3, len(augmentation_techniques)))
            selected_techniques = random.sample(augmentation_techniques, num_techniques)
            
            for technique in selected_techniques:
                if technique == 'gaussian_noise':
                    current_audio = self.add_gaussian_noise(current_audio)
                elif technique == 'time_stretch':
                    current_audio = self.time_stretch(current_audio)
                elif technique == 'pitch_shift':
                    current_audio = self.pitch_shift(current_audio, sr)
                elif technique == 'room_impulse':
                    current_audio = self.add_room_impulse_response(current_audio, sr)
                elif technique == 'compression':
                    current_audio = self.apply_dynamic_range_compression(current_audio)
            
            augmented_audios.append(current_audio)
        
        return augmented_audios
    
    def process_dataset(self, input_path, output_path, apply_noise_reduction=True):
        """
        Process entire dataset with augmentation and noise reduction
        
        Args:
            input_path (str): Path to input dataset
            output_path (str): Path to save augmented dataset
            apply_noise_reduction (bool): Whether to apply noise reduction
        """
        print("🚀 Starting Data Augmentation and Noise Handling Pipeline...")
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        
        # Create subdirectories for original and augmented data
        original_clean_path = os.path.join(output_path, 'original_clean')
        augmented_path = os.path.join(output_path, 'augmented')
        
        os.makedirs(original_clean_path, exist_ok=True)
        os.makedirs(augmented_path, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print("❌ No audio files found!")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        total_generated = 0
        
        for file_path in tqdm(audio_files, desc="Processing files"):
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # Apply noise reduction to original
                if apply_noise_reduction:
                    clean_audio = self.reduce_noise(audio, sr)
                else:
                    clean_audio = audio
                
                # Save cleaned original
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                clean_file_path = os.path.join(original_clean_path, f"{base_name}_clean.wav")
                sf.write(clean_file_path, clean_audio, sr)
                
                # Generate augmented versions
                augmented_audios = self.augment_single_file(clean_audio, sr)
                
                # Save augmented versions
                for idx, aug_audio in enumerate(augmented_audios):
                    aug_file_path = os.path.join(augmented_path, f"{base_name}_aug_{idx+1}.wav")
                    sf.write(aug_file_path, aug_audio, sr)
                    total_generated += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"✅ Processing complete!")
        print(f"📊 Generated {total_generated} augmented files")
        print(f"📁 Original clean files: {len(os.listdir(original_clean_path))}")
        print(f"📁 Augmented files: {len(os.listdir(augmented_path))}")
        
        # Generate processing report
        self._generate_augmentation_report(input_path, output_path, len(audio_files), total_generated)
    
    def _generate_augmentation_report(self, input_path, output_path, original_count, augmented_count):
        """Generate augmentation processing report"""
        report = f"""
# Data Augmentation and Noise Handling Report

## Processing Summary
- **Input Directory**: {input_path}
- **Output Directory**: {output_path}
- **Original Files**: {original_count}
- **Augmented Files Generated**: {augmented_count}
- **Augmentation Factor**: {self.augmentation_factor}
- **Total Dataset Size**: {original_count + augmented_count}

## Techniques Applied
1. **Noise Reduction**: {'Applied' if NOISEREDUCE_AVAILABLE else 'Spectral Subtraction'}
2. **Gaussian Noise Addition**: Various SNR levels (10-30 dB)
3. **Time Stretching**: Speed variation (0.8x - 1.2x)
4. **Pitch Shifting**: ±2 semitones
5. **Room Impulse Response**: Small room, large hall, garage simulation
6. **Dynamic Range Compression**: Threshold-based compression

## Output Structure
```
{output_path}/
├── original_clean/     # Noise-reduced original files
└── augmented/         # Augmented versions
```

## Recommendations
- Use augmented data for training to improve model generalization
- Validate on original clean data to assess real-world performance
- Consider balancing classes if augmentation creates imbalance
- Monitor for overfitting with heavily augmented data

Processing completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(os.path.join(output_path, 'augmentation_report.md'), 'w') as f:
            f.write(report)
        
        print("📋 Augmentation report generated!")

class NoiseReductionPipeline:
    """Standalone noise reduction pipeline"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def reduce_noise_dataset(self, input_path, output_path):
        """
        Apply noise reduction to entire dataset
        
        Args:
            input_path (str): Path to noisy dataset
            output_path (str): Path to save clean dataset
        """
        print("🧹 Starting Noise Reduction Pipeline...")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print("❌ No audio files found!")
            return
        
        for file_path in tqdm(audio_files, desc="Reducing noise"):
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # Apply noise reduction
                if NOISEREDUCE_AVAILABLE:
                    clean_audio = nr.reduce_noise(y=audio, sr=sr)
                else:
                    clean_audio = self._spectral_subtraction(audio, sr)
                
                # Save clean audio
                base_name = os.path.basename(file_path)
                clean_file_path = os.path.join(output_path, f"clean_{base_name}")
                sf.write(clean_file_path, clean_audio, sr)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print("✅ Noise reduction complete!")
    
    def _spectral_subtraction(self, audio, sr):
        """Apply spectral subtraction noise reduction"""
        # Implementation similar to the one in DataAugmentationPipeline
        try:
            noise_sample_length = int(0.5 * sr)
            if len(audio) > noise_sample_length:
                noise_sample = audio[:noise_sample_length]
            else:
                noise_sample = audio[:len(audio)//4]
            
            stft_audio = librosa.stft(audio)
            stft_noise = librosa.stft(noise_sample)
            
            noise_spectrum = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
            
            magnitude = np.abs(stft_audio)
            phase = np.angle(stft_audio)
            
            clean_magnitude = magnitude - 0.5 * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            print(f"Spectral subtraction failed: {e}")
            return audio

def main():
    """Main function to run data augmentation pipeline"""
    # Configuration
    INPUT_PATH = "./data/"  # Update with your dataset path
    OUTPUT_PATH = "./data/augmented_dataset/"
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input path '{INPUT_PATH}' not found!")
        print("Please update the INPUT_PATH variable with the correct path to your audio files.")
        return
    
    # Initialize augmentation pipeline
    aug_pipeline = DataAugmentationPipeline(
        sample_rate=22050,
        augmentation_factor=4  # Generate 4 augmented versions per original file
    )
    
    # Process dataset
    aug_pipeline.process_dataset(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        apply_noise_reduction=True
    )
    
    print("\n✅ Data augmentation and noise handling completed successfully!")
    print(f"📁 Check {OUTPUT_PATH} for processed files")
    
    # Optional: Run standalone noise reduction
    if input("\n🤔 Run additional noise reduction on original files? (y/n): ").lower() == 'y':
        noise_reducer = NoiseReductionPipeline()
        clean_output_path = "./data/clean_dataset/"
        noise_reducer.reduce_noise_dataset(INPUT_PATH, clean_output_path)

if __name__ == "__main__":
    main()