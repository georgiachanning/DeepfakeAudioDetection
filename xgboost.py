# Import necessary libraries for machine learning and data processing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from features import Features  # Custom module for feature generation
import librosa  # For audio processing
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from pickle import dump  # To save models
from imblearn.under_sampling import RandomUnderSampler  # To balance dataset classes
from azureml.fsspec import AzureMachineLearningFileSystem  # For working with AzureML file system
from itertools import product  # For creating cartesian product of parameters

# Class definition for the model that uses Gradient Boosting Classifier
class XGBooster(object):
    
    # Initialization method
    def __init__(self, to_save=True, \
            rng_seed=0, splitvoice=False, fakeavceleb=False, compressed=False, rerecorded=False, \
            max=40, train_only=False, slice_asvspoof=True, slice_size_seconds=2, balanced_classes=True,
            mode='KEY', n_estimators=100, max_depth=10, data_only=False, subset=None):
        
        # Initialize the feature generator using the provided subset of features
        self.feature_generator = Features(subset=subset)

        # Assign instance variables based on constructor parameters
        self.rng_seed = rng_seed  # Random seed for reproducibility
        self.to_save = to_save  # Flag to save models and results
        self.mode = mode  # Mode of operation (e.g., 'KEY', 'splitvoice')
        self.train_only = train_only  # Whether to train only without testing
        self.slice_asvspoof = slice_asvspoof  # Whether to slice audio for ASVspoof dataset
        self.slice_size = slice_size_seconds  # Size of the audio slice in seconds
        self.max = max  # Maximum number of samples to use
        self.subset = subset  # Subset of features to use
        self.balanced_classes = balanced_classes  # Whether to balance classes during training

        # Load different datasets based on provided flags
        if splitvoice:
            self.load_splitvoice_data()
            self.mode = 'splitvoice'
        elif fakeavceleb:
            self.load_fakeavceleb_data()
            self.mode = 'fakeavceleb'
        elif compressed:
            self.load_compressed_asvspoof_data()
        elif rerecorded:
            self.load_rerecorded_asvspoof_data()
        else:
            self.load_asvspoof_data()

        # If only data is needed, skip model training
        if not data_only:
            # Set parameters for the Gradient Boosting Classifier
            self.n_estimators = n_estimators
            self.max_depth = max_depth

            # If max_depth and n_estimators are integers, train the model with these values
            if isinstance(max_depth, int) and isinstance(n_estimators, int):
                self.booster = GradientBoostingClassifier(random_state=rng_seed, n_estimators=self.n_estimators, max_depth=self.max_depth)

                # Train and test the model
                self.train()
                self.test()

                # Save the model if required
                if self.to_save:
                    self.save()
            else:
                # If max_depth and n_estimators are lists, perform grid search by training multiple models
                experiment_results = []
                configs = list(product(max_depth, n_estimators))  # Get all combinations of max_depth and n_estimators
                for c in configs:
                    exp = {}
                    exp['num_datapoints'] = self.max  # Track number of data points
                    exp['max_depth'] = c[0]
                    exp['num_estimators'] = c[1]
                    exp['task'] = self.mode

                    # Set max_depth and n_estimators for each experiment
                    self.max_depth = c[0]
                    self.n_estimators = c[1]
                    print("Now training with {} estimators and {} max depth.".format(self.n_estimators, self.max_depth))

                    # Train the model
                    self.booster = GradientBoostingClassifier(random_state=rng_seed, n_estimators=self.n_estimators, max_depth=self.max_depth)
                    self.train()

                    # Test the model and record results
                    y_pred = self.test()
                    if self.to_save:
                        self.save()

                    exp['accuracy'] = accuracy_score(self.y_test, y_pred)
                    experiment_results.append(exp)
                
                # Save results of all experiments to a CSV file
                res = pd.DataFrame(experiment_results)
                if self.to_save:
                    res.to_csv('./boosters/booster_exp_{}_{}_{}_{}_rerecorded.csv'.format(str(self.max), str(self.slice_size), self.mode, str(self.rng_seed)), index=False)

    # Load FakeAVCeleb dataset
    def load_fakeavceleb_data(self):
        self.train_dir = "azureml:"
        fs = AzureMachineLearningFileSystem(self.train_dir)

        train_features = []
        metadata_file = f"{self.train_dir}/metadata.csv"
        metadata = pd.read_csv(metadata_file)
        self.filenames = [f"{self.train_dir}/{file}" for file in metadata['new_filename'].to_list()]
        self.labels = metadata['category'].to_list()

        # Extract audio features for each file
        for file in self.filenames:
            with fs.open(file, 'r') as f:
                segment, sr = librosa.load(f)
                if self.slice_size != None:
                    segment = segment[:int(sr*self.slice_size)]
                train_features.append(self.feature_generator.make_features(segment, sr))

        self.X_train = np.array(train_features)
        self.y_train = np.array(self.labels)
        print("loaded train audio, y_train contains {} samples".format(len(self.y_train)))

        # Split dataset into training and testing sets
        if not self.train_only:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(self.X_train), np.array(self.y_train), test_size=0.33, random_state=self.rng_seed)
        else:
            print("skipping loading test audio")
        return

    # Other similar functions for loading datasets (splitvoice, asvspoof, compressed data, etc.) are defined below...
    
    # Save the trained model to disk
    def save(self):
        print("saving now")
        filename = './boosters/booster_audio_len_{}_max_depth_{}_n_est_{}_data_max_{}_mode_{}_rerecorded.pkl'.format(self.slice_size, self.max_depth, self.n_estimators, self.max, self.mode)
        with open(filename, "wb") as f:
            dump(self.booster, f, protocol=5)
        return

    # Train the model using the training data
    def train(self):
        self.booster.fit(self.X_train, self.y_train)
        return

    # Test the model using the test data and print classification report
    def test(self):
        y_pred = self.booster.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, digits=4))

        # Save the classification report if required
        filename = './boosters/booster_audio_len_{}_max_depth_{}_n_est_{}_data_max_{}_mode_{}_rerecorded.txt'.format(self.slice_size, self.max_depth, self.n_estimators, self.max, self.mode)
        if self.to_save:
            with open(filename, "w") as f:
                f.write(classification_report(self.y_test, y_pred))

        return y_pred
    
# Main function to initialize and run the model
if __name__=='__main__':
    boost = XGBooster(max=100000, slice_size_seconds=6, splitvoice=False, fakeavceleb=False, rerecorded=False,  n_estimators=[400], max_depth=[8], \
                      balanced_classes=True, subset=['mfcc_20', 'mfcc_4', 'rms'])
