"""
Diagnostic: Check if you have 8,270 or 16,540 training images

Run this to understand what your preprocessing saved.
"""

import numpy as np
import os
import sys

def check_preprocessing_data(data_dir='preprocessed_data/sub-02'):
    """Check what preprocessing data exists and its shape"""
    
    print("="*70)
    print("PREPROCESSING DATA DIAGNOSTIC")
    print("="*70)
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Directory not found: {data_dir}")
        print("\nPlease provide the correct path to your preprocessed data:")
        print("  python check_preprocessing.py --data_dir /path/to/preprocessed_data/sub-02")
        return
    
    print(f"\nChecking directory: {data_dir}")
    print(f"\nFiles found:")
    
    files = {}
    for f in os.listdir(data_dir):
        if f.endswith('.npy'):
            path = os.path.join(data_dir, f)
            loaded = np.load(path, allow_pickle=True)
            # Extract dictionary from object array
            data_dict = loaded.item()
            # Get the actual EEG data shape
            eeg_data = data_dict['preprocessed_eeg_data']
            files[f] = eeg_data.shape
            print(f"  {f}: {eeg_data.shape}")
    
    if not files:
        print("  ❌ No .npy files found!")
        return
    
    # Check main training file
    print("\n" + "="*70)
    print("TRAINING DATA CHECK")
    print("="*70)
    
    training_files = [f for f in files.keys() if 'training' in f.lower()]
    
    if not training_files:
        print("\n❌ No training file found!")
        print("   Expected: preprocessed_eeg_training.npy")
        return
    
    main_file = training_files[0]
    train_shape = files[main_file]
    
    print(f"\nMain training file: {main_file}")
    print(f"Shape: {train_shape}")
    
    num_images = train_shape[0]
    num_reps = train_shape[1] if len(train_shape) > 1 else 1
    
    print(f"\nParsing shape:")
    print(f"  Number of images: {num_images}")
    print(f"  Repetitions per image: {num_reps}")
    
    # Determine status
    print("\n" + "="*70)
    print("STATUS")
    print("="*70)
    
    if num_images == 1654:
        print("\n❌ CONCEPT-LEVEL DATA (1,654 concepts)")
        print("   This is why you're getting 0.5% test accuracy!")
        print("   You averaged across the 10 images per concept.")
        print("\n   SOLUTION: Re-run preprocessing WITHOUT averaging images.")
        
    elif num_images == 8270:
        print("\n⚠️  IMAGE-LEVEL DATA - BUT ONLY HALF! (8,270 images)")
        print("   You have image-level data (good!) but only one subset.")
        print("   Missing: The other 8,270 images from the other subset.")
        print(f"\n   Expected total: 16,540 images")
        print(f"   You have: {num_images} images")
        print(f"   Missing: {16540 - num_images} images")
        print("\n   SOLUTION: Combine all 4 sessions to get 16,540 images.")
        print("   See: COMBINING_SESSIONS_TO_GET_16540.md")
        
    elif num_images == 16540:
        print("\n✅ FULL IMAGE-LEVEL DATA! (16,540 images)")
        print("   You have all training images!")
        print("   This is the correct preprocessing.")
        print("\n   With this data, you should achieve:")
        print("     - Validation: 20-40%")
        print("     - Test: 10-25% (ENIGMA baseline is 27.6%)")
        
    else:
        print(f"\n❓ UNEXPECTED IMAGE COUNT: {num_images}")
        print(f"   Expected: 1,654 (concept), 8,270 (half), or 16,540 (full)")
        print(f"   Got: {num_images}")
        print("\n   Please check your preprocessing script.")
    
    # Check for session files
    print("\n" + "="*70)
    print("SESSION FILES CHECK")
    print("="*70)
    
    session_files = [f for f in files.keys() if 'session' in f.lower()]
    
    if session_files:
        print(f"\n✓ Found {len(session_files)} session files:")
        for sf in sorted(session_files):
            print(f"  {sf}: {files[sf]}")
        
        print("\n💡 You can combine these session files to get all 16,540 images!")
        print("   Use the script: combine_sessions.py")
    else:
        print("\n⚠️  No individual session files found.")
        print("   If you need 16,540 images, you'll need to re-run preprocessing.")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if num_images == 1654:
        print("\n1. Re-run preprocessing WITHOUT averaging the 10 images per concept")
        print("2. Expected output: (16540, reps, 17, 250)")
        print("3. This should improve test accuracy from 0.5% to 10-20%")
        
    elif num_images == 8270:
        if session_files:
            print("\n1. Run: python combine_sessions.py")
            print("   This will combine your session files into 16,540 images")
            print("2. Expected output: (16540, reps, 17, 250)")
            print("3. This should improve test accuracy from 1% to 10-20%")
        else:
            print("\n1. Re-run preprocessing to save all 4 sessions")
            print("2. Make sure to combine sessions 1&3 and 2&4")
            print("3. Expected output: (16540, reps, 17, 250)")
            print("4. This should improve test accuracy from 1% to 10-20%")
    
    elif num_images == 16540:
        print("\n1. Your preprocessing is correct!")
        print("2. Train with standard hyperparameters:")
        print("   python train_new_preprocessing.py --subject 2")
        print("3. Expected test accuracy: 10-25%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='preprocessed_data/sub-02',
                       help='Path to preprocessed data directory')
    args = parser.parse_args()
    
    check_preprocessing_data(args.data_dir)