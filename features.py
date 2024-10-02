# Import necessary libraries for audio processing and file handling
import os  # To interact with the file system
import re  # Regular expressions for pattern matching
import numpy as np  # Numerical operations with arrays
import random  # For random operations
from pydub import AudioSegment  # For manipulating audio files
import math  # Math operations
import json  # For reading and writing JSON data
import librosa  # Audio analysis and feature extraction
from azureml.fsspec import AzureMachineLearningFileSystem  # For working with AzureML file system

# Define the Features class to extract different audio features from audio signals
class Features(object):
    def __init__(self, mode='traditional', subset=None):
        """
        Initialize the Features class.
        
        Args:
        - mode: The mode for extracting features ('traditional' by default).
        - subset: A list of specific features to extract, if provided.
        """
        self.mode = mode
        self.subset = subset

    # Method to extract MFCC (Mel-frequency cepstral coefficients) features from audio
    def mfcc(self, y, sr):
        return librosa.feature.mfcc(y=y, sr=sr)
    
    # Method to calculate zero-crossing rate (the rate at which the signal changes sign)
    def crossing_rate(self, y, sr=0):
        return librosa.feature.zero_crossing_rate(y=y)
    
    # Method to calculate the chromagram from a waveform (used for pitch-related analysis)
    def chromagram(self, y, sr):
        return librosa.feature.chroma_stft(y=y, sr=sr)

    # Method to calculate spectral centroid (the center of mass of the spectrum)
    def spectral_centroid(self, y, sr):
        return librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Method to calculate spectral bandwidth (the spread of the spectrum)
    def spectral_bandwidth(self, y, sr):
        return librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    # Method to calculate spectral rolloff (the frequency below which most of the energy is concentrated)
    def rolloff(self, y, sr):
        return librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # Method to calculate the root mean square (RMS) energy of the audio signal
    def root_mean_square(self, y, sr=0):
        return librosa.feature.rms(y=y)
    
    # Method to create features from audio based on the subset or extract all if no subset is provided
    def make_features(self, y, sr):
        """
        Extract audio features based on either a predefined subset or a full set of features.
        
        Args:
        - y: The audio time series.
        - sr: The sampling rate of the audio signal.

        Returns:
        - Flattened array of extracted audio features.
        """
        if self.subset is None:  # Extract all features if no specific subset is defined
            mfcc = np.mean(self.mfcc(y, sr).T, axis=0)
            cr = np.mean(self.crossing_rate(y, sr), axis=1)
            chroma = np.mean(self.chromagram(y, sr), axis=1)
            sc = np.mean(self.spectral_centroid(y, sr), axis=1)
            sb = np.mean(self.spectral_bandwidth(y, sr), axis=1)
            rolloff = np.mean(self.rolloff(y, sr), axis=1)
            rms = np.mean(self.root_mean_square(y, sr), axis=1)
            # Concatenate all features into a single feature vector
            features = np.concatenate((mfcc, cr, chroma, sc, sb, rolloff, rms))
        else:  # Extract only the features specified in the subset
            feats = []
            for func in self.subset:
                if 'mfcc' in func:
                    if '_' in func:  # If a specific MFCC coefficient is requested
                        idx = int(func.split('_')[1]) - 1
                        mfcc = np.mean(self.mfcc(y, sr).T, axis=0)[idx]
                    else:  # If all MFCCs are requested
                        mfcc = np.mean(self.mfcc(y, sr).T, axis=0)
                    feats.append(mfcc)
                elif func == 'cr':  # Zero-crossing rate
                    cr = np.mean(self.crossing_rate(y, sr), axis=1)
                    feats.append(cr)
                elif 'chroma' in func:  # Chromagram
                    if '_' in func:
                        idx = int(func.split('_')[1]) - 1
                        chroma = np.mean(self.chromagram(y, sr), axis=1)[idx]
                    else:
                        chroma = np.mean(self.chromagram(y, sr), axis=1)
                    feats.append(chroma)
                elif func == 'sc':  # Spectral centroid
                    sc = np.mean(self.spectral_centroid(y, sr), axis=1)
                    feats.append(sc)
                elif func == 'sb':  # Spectral bandwidth
                    sb = np.mean(self.spectral_bandwidth(y, sr), axis=1)
                    feats.append(sb)
                elif func == 'rolloff':  # Spectral rolloff
                    rolloff = np.mean(self.rolloff(y, sr), axis=1)
                    feats.append(rolloff)
                elif func == 'rms':  # Root mean square (RMS)
                    rms = np.mean(self.root_mean_square(y, sr), axis=1)
                    feats.append(rms)
                else:
                    # Raise an error if an unknown feature is requested
                    raise KeyError("Don't recognize this function.")
            # Convert the list of features to a NumPy array
            features = np.array(feats)
        # Return a flattened feature array
        return features.flatten()

# Function to compress audio files in a given directory and save them in another directory as MP3 files
def compress(fs, in_dir, out_dir):
    """
    Compress audio files from one directory and save them as MP3 files in another directory.
    
    Args:
    - fs: File system handler.
    - in_dir: Input directory containing audio files.
    - out_dir: Output directory to save compressed MP3 files.
    """
    for file in fs.ls(in_dir):  # List all files in the input directory
        fn = os.path.join(in_dir, file.split('/')[-1])  # Extract the file name
        with fs.open(fn, 'r') as f:
            # Convert the audio file to MP3 format with 128kbps bitrate
            AudioSegment.from_file(f, 'flac').export(os.path.join(out_dir, file.split('/')[-1]), format="mp3", bitrate="128k")
    return

# Main block to create an instance of the Features class
if __name__ == '__main__':
    Features()  # Initialize the Features class
