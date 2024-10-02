# Import necessary libraries for file handling, audio processing, and other utilities
import os  # For file and directory operations
import re  # For regular expression matching
import numpy as np  # For numerical operations
import random  # For random sampling and operations
from pydub import AudioSegment  # For audio file manipulation and processing
import math  # Mathematical operations
import json  # For reading and writing JSON data
import librosa  # For audio processing
from azureml.fsspec import AzureMachineLearningFileSystem  # For working with AzureML file system

# Class to preprocess real and fake audio data
class Preprocess(object):
    def __init__(self, src_dir_real, src_dir_fake, subset='real', audio_length=3, out_dir='./data/3sec_shortened'):
        """
        Initialize the Preprocess class.

        Args:
        - src_dir_real: Directory containing real audio samples.
        - src_dir_fake: Directory containing fake audio samples.
        - subset: Type of subset (default is 'real').
        - audio_length: Length of each audio segment in seconds.
        - out_dir: Output directory for preprocessed audio files.
        """
        self.audio_length = audio_length  # Audio length in seconds
        self.num_milliseconds = int(audio_length * 1000)  # Convert audio length to milliseconds (Pydub works in ms)
        self.src_dir_real = src_dir_real  # Source directory for real audio
        self.src_dir_fake = src_dir_fake  # Source directory for fake audio
        self.out_dir = out_dir  # Output directory for processed audio

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Call the preprocessing method
        self.preprocess()

    # Method to preprocess real and fake audio files
    def preprocess(self):
        """
        Preprocess real and fake audio files by segmenting them into smaller chunks and saving them.
        """
        # Process real audio files
        all_files = [os.path.join(self.src_dir_real, file) for file in os.listdir(self.src_dir_real)]
        for file in all_files:
            wav = AudioSegment.from_wav(file)  # Load the audio file
            index = 0
            # Split the audio file into smaller segments
            for sample in range(0, len(wav) - self.num_milliseconds, self.num_milliseconds):
                subsample = wav[sample:sample + self.num_milliseconds]  # Extract a segment of the audio
                label = 'real' if 'original' in file else 'fake'  # Assign label based on file name
                # Construct the output file path and name
                outfile = '{}/{}/{}/{}_{}.wav'.format(self.out_dir, 
                                                      label, 
                                                      file.split('/')[-1].lower().split('-')[0],
                                                      file.split('/')[-1].strip('.wav').lower(), 
                                                      str(index))
                print('real: ', outfile)
                # Create output directories if they don't exist
                if not os.path.exists('/'.join(outfile.split('/')[:-1])):
                    os.makedirs('/'.join(outfile.split('/')[:-1]))
                subsample.export(outfile, format='wav')  # Save the audio segment as a WAV file
                index += 1

        # Process fake audio files
        all_files = [os.path.join(self.src_dir_fake, file) for file in os.listdir(self.src_dir_fake)]
        for file in all_files:
            wav = AudioSegment.from_wav(file)
            index = 0
            # Split the fake audio file into segments
            for sample in range(0, len(wav) - self.num_milliseconds, self.num_milliseconds):
                subsample = wav[sample:sample + self.num_milliseconds]
                label = 'real' if 'real' in file else 'fake'  # Assign label based on file name
                # Construct the output file path and name
                outfile = '{}/{}/sounds_like_{}/{}_{}.wav'.format(self.out_dir, 
                                                                  label, 
                                                                  re.search(r'to-(.*?)\.wav', file.lower()).group(1),
                                                                  file.split('/')[-1].strip('.wav').lower(), 
                                                                  str(index))
                print('fake: ', outfile)
                # Create output directories if they don't exist
                if not os.path.exists('/'.join(outfile.split('/')[:-1])):
                    os.makedirs('/'.join(outfile.split('/')[:-1]))
                subsample.export(outfile, format='wav')  # Save the audio segment as a WAV file
                index += 1
        return

# Class to mix real and fake audio files to create hybrid (mixed) audio samples
class Mix(object):
    def __init__(self, src='./data/3sec_shortened', dest='./data/3sec_mixed', splice_length=0.3, rng_seed=0):
        """
        Initialize the Mix class.

        Args:
        - src: Source directory containing preprocessed real and fake audio files.
        - dest: Destination directory to save mixed audio files.
        - splice_length: Length of the splice segment to mix (in seconds).
        - rng_seed: Seed for random operations to ensure reproducibility.
        """
        self.src_path = src  # Source directory for preprocessed audio
        self.dest_path = dest  # Destination directory for mixed audio
        self.splice_length = splice_length  # Splice length in seconds
        self.num_milliseconds = splice_length * 1000  # Convert splice length to milliseconds

        # Set random seeds for reproducibility
        np.random.seed(rng_seed)
        random.seed(rng_seed)

        # Create output directories if they don't exist
        if not os.path.exists(os.path.join(self.dest_path, 'real')):
            os.makedirs(os.path.join(self.dest_path, 'real'))
        if not os.path.exists(os.path.join(self.dest_path, 'mixed')):
            os.makedirs(os.path.join(self.dest_path, 'mixed'))
        if not os.path.exists(os.path.join(self.dest_path, 'fake')):
            os.makedirs(os.path.join(self.dest_path, 'fake'))

        # Call the mixing method
        self.mix()

    # Method to mix real and fake audio segments to create a hybrid
    def mix_real_and_fake(self, real_file, fake_file):
        """
        Mix a real and a fake audio file to create a hybrid (mixed) audio file.

        Args:
        - real_file: Path to the real audio file.
        - fake_file: Path to the fake audio file.

        Returns:
        - mixed_audio: The resulting mixed audio segment.
        - fake_interval: The interval where the fake audio was spliced into the real audio.
        """
        fake_audio = AudioSegment.from_wav(fake_file)  # Load the fake audio
        real_audio = AudioSegment.from_wav(real_file)  # Load the real audio

        # Randomly choose a splice start point in both real and fake audio
        splice_start_fake = np.random.randint(0, len(fake_audio) - self.num_milliseconds + 1)
        splice_start_real = np.random.randint(0, len(real_audio) - self.num_milliseconds + 1)

        # Create the mixed audio by replacing a segment of real audio with fake audio
        mixed_audio = real_audio[:splice_start_real] + \
            fake_audio[splice_start_fake:splice_start_fake + self.num_milliseconds] + \
            real_audio[splice_start_real + self.num_milliseconds:]
        fake_interval = (splice_start_real, self.num_milliseconds)  # Keep track of where the splice happened

        return mixed_audio, fake_interval

    # Method to perform the entire mixing process for all audio files
    def mix(self):
        """
        Mix real and fake audio samples and save them as mixed audio files in the destination directory.
        """
        # Loop through voices (categories) in the real directory
        for voice in os.listdir(os.path.join(self.src_path, 'real')):
            real_samples = os.listdir(os.path.join(self.src_path, 'real', voice))
            random.shuffle(real_samples)  # Shuffle the real samples
            split_index = 100  # Set split index (first 100 samples for real, the rest for mixing)
            keep_real = real_samples[:split_index]  # Keep some real samples unchanged
            make_mix_real = real_samples[split_index:200]  # Use some real samples for mixing

            # Copy real tracks directly to the destination
            for real in keep_real:
                os.system('cp {} {}'.format(os.path.join(self.src_path, 'real', voice, real), os.path.join(self.dest_path, 'real', real)))

            # Process fake samples for the same voice
            fake_samples = os.listdir(os.path.join(self.src_path, 'fake', 'sounds_like_' + voice))
            random.shuffle(fake_samples)  # Shuffle the fake samples
            keep_fake = fake_samples[:split_index]  # Keep some fake samples unchanged
            make_mix_fake = fake_samples[split_index:200]  # Use some fake samples for mixing

            # Copy fake tracks directly to the destination
            for fake in keep_fake:
                os.system('cp {} {}'.format(os.path.join(self.src_path, 'fake', 'sounds_like_' + voice, fake), os.path.join(self.dest_path, 'fake', fake)))

            # Create a list to store information about the mixed audio files
            mixed_dicts = []
            # Loop through the real samples to be mixed and mix them with corresponding fake samples
            for index, real in enumerate(make_mix_real):
                mixed_dict = {}
                real_file = os.path.join(self.src_path, 'real', voice, real)
                fake_file = os.path.join(self.src_path, 'fake', 'sounds_like_' + voice, make_mix_fake[index])

                # Store file paths in the dictionary
                mixed_dict['real_audio'] = real_file
                mixed_dict['fake_audio'] = fake_file
                # Mix the real and fake audio files
                mixed_audio, interval = self.mix_real_and_fake(real_file=real_file, fake_file=fake_file)
                mixed_file = os.path.join(self.dest_path, 'mixed', voice + '_mixed_' + str(index) + '.wav')
                mixed_dict['mixed_audio'] = mixed_file
                mixed_dict['interval'] = interval
                mixed_dicts.append(mixed_dict)

                # Save the mixed audio file
                mixed_audio.export(mixed_file, format='wav')

        # Save the dictionary of mixed audio information as a JSON file
        with open(os.path.join(self.dest_path, 'mixed', 'mixed.json'), 'w') as fp:
            json.dump(mixed_dicts, fp)

        return
