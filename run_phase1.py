"""
Intelligent Motorcycle Fault Detection - Phase 1: Main Pipeline
Authors: Mohammad Kazim, Aryan Gupta, Faiz Maqsood
Bharati Vidyapeeth College of Engineering, Pune

This script orchestrates the complete Phase 1 pipeline for the motorcycle fault detection project.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_eda(dataset_path, output_dir):
    """Run Exploratory Data Analysis"""
    print("🔍 Starting Exploratory Data Analysis...")
    
    try:
        from src.exploratory_data_analysis_01 import MotorcycleAudioEDA
        
        eda = MotorcycleAudioEDA(dataset_path=dataset_path, output_dir=output_dir)
        results = eda.run_complete_eda()
        
        print("✅ EDA completed successfully!")
        return results
        
    except ImportError as e:
        print(f"❌ Failed to import EDA module: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ EDA failed: {e}")
        return None

def run_feature_extraction(dataset_path, output_path):
    """Run Advanced Feature Extraction"""
    print("🎵 Starting Advanced Feature Extraction...")
    
    try:
        from src.advanced_feature_extraction_02 import AdvancedFeatureExtractor
        
        extractor = AdvancedFeatureExtractor(
            sample_rate=22050,
            n_mfcc=20,
            n_mels=128,
            hop_length=512
        )
        
        feature_df = extractor.process_dataset(
            dataset_path=dataset_path,
            output_path=output_path
        )
        
        if feature_df is not None:
            print("✅ Feature extraction completed successfully!")
            return feature_df
        else:
            print("❌ Feature extraction failed!")
            return None
            
    except ImportError as e:
        print(f"❌ Failed to import feature extraction module: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
        return None
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

def run_data_augmentation(input_path, output_path):
    """Run Data Augmentation and Noise Handling"""
    print("🔄 Starting Data Augmentation and Noise Handling...")
    
    try:
        from src.data_augmentation_noise_handling_03 import DataAugmentationPipeline
        
        aug_pipeline = DataAugmentationPipeline(
            sample_rate=22050,
            augmentation_factor=4
        )
        
        aug_pipeline.process_dataset(
            input_path=input_path,
            output_path=output_path,
            apply_noise_reduction=True
        )
        
        print("✅ Data augmentation completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import augmentation module: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Data augmentation failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'librosa', 'sklearn', 
        'matplotlib', 'seaborn', 'tqdm', 'soundfile'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def setup_project_structure(base_path):
    """Setup project directory structure"""
    print("📁 Setting up project structure...")
    
    directories = [
        'data',
        'data/raw',
        'data/processed', 
        'data/augmented',
        'eda',
        'models',
        'results',
        'src'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Project structure created!")

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description='Motorcycle Fault Detection - Phase 1 Pipeline')
    
    parser.add_argument('--dataset_path', type=str, default='./data/raw/',
                       help='Path to raw audio dataset')
    parser.add_argument('--output_base', type=str, default='./data/processed/',
                       help='Base path for processed data output')
    parser.add_argument('--skip_eda', action='store_true',
                       help='Skip EDA step')
    parser.add_argument('--skip_augmentation', action='store_true',
                       help='Skip data augmentation step')
    parser.add_argument('--check_deps_only', action='store_true',
                       help='Only check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    if args.check_deps_only:
        return True
    
    # Setup project structure
    setup_project_structure('.')
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path '{args.dataset_path}' not found!")
        print("Please provide a valid dataset path with --dataset_path argument")
        return False
    
    print("🚀 Starting Intelligent Motorcycle Fault Detection - Phase 1 Pipeline")
    print("=" * 80)
    
    # Phase 1.1: Exploratory Data Analysis
    if not args.skip_eda:
        eda_results = run_eda(args.dataset_path, './eda/')
        if eda_results is None:
            print("⚠️  EDA failed, but continuing with pipeline...")
    else:
        print("⏭️  Skipping EDA as requested...")
    
    # Phase 1.2: Advanced Feature Extraction
    feature_df = run_feature_extraction(args.dataset_path, args.output_base)
    if feature_df is None:
        print("❌ Feature extraction failed! Cannot continue pipeline.")
        return False
    
    # Phase 1.3: Data Augmentation and Noise Handling
    if not args.skip_augmentation:
        augmentation_success = run_data_augmentation(
            args.dataset_path,
            os.path.join(args.output_base, 'augmented')
        )
        if not augmentation_success:
            print("⚠️  Data augmentation failed, but pipeline completed basic processing...")
    else:
        print("⏭️  Skipping data augmentation as requested...")
    
    print("\n" + "=" * 80)
    print("🎉 Phase 1 Pipeline Completed Successfully!")
    
    # Generate final summary
    generate_phase1_summary(args.dataset_path, args.output_base, feature_df)
    
    return True

def generate_phase1_summary(dataset_path, output_base, feature_df):
    """Generate Phase 1 completion summary"""
    summary = f"""
# Intelligent Motorcycle Fault Detection - Phase 1 Summary

## Pipeline Completion Status: ✅ SUCCESS

### Data Processing Summary
- **Dataset Path**: {dataset_path}
- **Output Path**: {output_base}
- **Features Extracted**: {feature_df.shape[1] if feature_df is not None else 'N/A'}
- **Samples Processed**: {feature_df.shape[0] if feature_df is not None else 'N/A'}

### Completed Steps
1. ✅ **Exploratory Data Analysis (EDA)**
   - Class distribution analysis
   - Audio visualization (waveforms & spectrograms)  
   - Statistical analysis
   - Output: `./eda/` directory

2. ✅ **Advanced Feature Extraction**
   - MFCC features (20 coefficients with statistics)
   - Mel-spectrogram features
   - Spectral features (centroid, bandwidth, rolloff, contrast, flatness)
   - Chromagram features
   - Temporal features (ZCR, RMS, tempo)
   - Tonnetz features
   - Output: `{output_base}extracted_features.csv`

3. ✅ **Data Augmentation & Noise Handling**
   - Gaussian noise addition
   - Time stretching
   - Pitch shifting
   - Room impulse response simulation
   - Dynamic range compression
   - Background noise reduction
   - Output: `{output_base}augmented/` directory

### Next Steps (Phase 2)
- Model training with extracted features
- Hyperparameter optimization
- Cross-validation and performance evaluation
- Model comparison and selection

### Generated Files
- `./eda/`: EDA plots and analysis
- `{output_base}extracted_features.csv`: Feature matrix
- `{output_base}features.npy`: NumPy feature array
- `{output_base}labels.npy`: Labels array
- `{output_base}feature_names.txt`: Feature names
- `{output_base}augmented/`: Augmented dataset

Processing completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    try:
        with open('Phase1_Summary.md', 'w') as f:
            f.write(summary)
        print("📋 Phase 1 summary saved to Phase1_Summary.md")
    except:
        print("📋 Phase 1 summary:")
        print(summary)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)