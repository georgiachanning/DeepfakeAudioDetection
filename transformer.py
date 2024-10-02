# Import necessary libraries and modules for working with transformers, datasets, and machine learning models
import transformers
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer, AutoFeatureExtractor, ASTForAudioClassification, EncodecModel
from datasets import load_dataset, Dataset, Audio
import torch  # PyTorch for tensor operations and deep learning
import pandas as pd  # Data manipulation
import os  # OS module for file handling
import random  # Random operations for shuffling
import numpy as np  # Numerical operations
import librosa  # For audio processing
from transformers.utils import logging  # Logging for transformers
import datasets  # Hugging Face's datasets library
from datasets import ClassLabel  # Handling class labels for classification
from azure.ai.ml import MLClient  # Azure Machine Learning Client for cloud services
from azureml.fsspec import AzureMachineLearningFileSystem  # AzureML file system for loading datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # Metrics for evaluating classification performance
import argparse  # Argument parser for command-line options

# Empty cache for PyTorch to free up GPU memory
torch.cuda.empty_cache()
# Set environment variable to allow PyTorch to allocate expandable CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Argument parsing to handle command-line arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=str, help="Path to input data")
parser.add_argument("--dataset_name", required=True, type=str, default='ASVSpoof', help="Name of the dataset to use")
parser.add_argument("--max_duration", required=True, type=float, help="Maximum duration of audio samples in seconds")
parser.add_argument("--model_name", required=True, type=str, help="Name of the pre-trained model to use")
parser.add_argument("--model_path", required=True, type=str, help="Path to the pre-trained model")
parser.add_argument("--output_path", required=False, default='./output', type=str, help="Directory to save the trained model")
parser.add_argument("--compressed", required=False, default=False, type=bool, help="Use compressed data")
parser.add_argument("--rerecorded", required=False, default=False, type=bool, help="Use re-recorded data")
parser.add_argument("--batch_size", required=False, default=4, type=int, help="Batch size for training")
parser.add_argument("--num_epochs", required=False, default=10, type=int, help="Number of epochs for training")
parser.add_argument("--max_size", required=False, default=100000, type=int, help="Maximum dataset size")

# Parse the arguments from the command line
args = parser.parse_args()

# Initialize Equal Error Rate (EER) flag, which might be used later in the script
eer = True

# Class to handle pre-trained transformer models for audio classification tasks
class PreTrainedTransformer(object):
    def __init__(self, model="MIT/ast-finetuned-audioset-10-10-0.4593", num_epochs=20, \
                dataset_name="ASVSpoof", dataset_path=None, max_size=1000, classification_type='KEY', \
                max_duration=3, compressed=False, rerecorded=False, data_only=False, \
                batch_size=32, output_dir='.'):
        
        # Initialize model checkpoint, feature extractor, and dataset configurations
        self.model_checkpoint = model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_checkpoint)

        self.dataset_path = dataset_path  # Path to the dataset
        self.batch_size = batch_size  # Batch size for training
        self.max_duration = max_duration  # Maximum duration for audio samples
        self.num_epochs = num_epochs  # Number of epochs for training
        self.classification_type = classification_type  # Type of classification task (e.g., binary or multi-class)
        self.max_size = max_size  # Maximum number of samples to use
        self.compressed = compressed  # Flag to indicate whether compressed data is used
        self.rerecorded = rerecorded  # Flag to indicate whether re-recorded data is used
        self.output_dir = output_dir  # Directory to store output models

        # Logging configuration for transformers
        logging.set_verbosity_error()

        # Logging configuration to record model and training settings
        config = {'dataset_size': max_size, 'max_duration': self.max_duration, 'model_base': self.model_checkpoint, 'num_epochs': self.num_epochs,
                  'classification_type': classification_type, 'dataset': dataset_name, 'batch_size': self.batch_size}
        
        # Initialize various evaluation metrics (e.g., accuracy, precision, recall, F1)
        self.metrics = {
            'accuracy': datasets.load_metric("accuracy", trust_remote_code=True),
            'precision': datasets.load_metric("precision", trust_remote_code=True),
            'recall': datasets.load_metric("recall", trust_remote_code=True),
            'f1': datasets.load_metric('f1', trust_remote_code=True)
        }

        self.dataset_name = dataset_name

        # Load the appropriate dataset based on the dataset_name provided (e.g., ASVSpoof, DeepVoice, FakeAVCeleb)
        if self.dataset_name == "ASVSpoof":
            self.dataset = ASVSpoofDataset(root_dir=self.dataset_path, max_size=max_size, classification_type=self.classification_type, sr=self.feature_extractor.sampling_rate, compressed=self.compressed, rerecorded=self.rerecorded).load_data()
        elif "DeepVoice" in self.dataset_name:
            if "mixed" in self.dataset_name:
                self.dataset = DeepVoiceDataset(mixed=True, root_dir=self.dataset_path, max_size=max_size, sr=self.feature_extractor.sampling_rate).load_data()
            else:
                self.dataset = DeepVoiceDataset(root_dir=self.dataset_path, max_size=max_size, sr=self.feature_extractor.sampling_rate).load_data()
        elif self.dataset_name == 'FakeAVCeleb':
            self.dataset = FakeAVCeleb(root_dir=self.dataset_path, max_size=max_size, sr=self.feature_extractor.sampling_rate).load_data()
        else:
            raise KeyError("Options are 'ASVSpoof', 'DeepVoice', or 'FakeAVCeleb'.")

        # Split dataset into training and testing sets
        self.dataset = self.dataset.train_test_split(test_size=0.33, shuffle=True)
        self.dataset_size = len(self.dataset)

        # Map label names to numeric IDs for classification tasks
        self.labels = self.dataset["train"].features["label"].names
        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label

        # If only data loading is requested, skip fine-tuning
        if data_only is False:
            self.finetune()

    # Preprocessing function to prepare audio samples for the model
    def preprocess_function(self, examples):
        audio_arrays = [x["array"][:int(self.feature_extractor.sampling_rate * self.max_duration)] for x in examples["audio"]]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
        )
        return inputs

    # Fine-tune the pre-trained model on the specific dataset
    def finetune(self):

        # Preprocess and encode the dataset
        self.encoded_dataset = self.dataset.map(self.preprocess_function, remove_columns=["audio", "filename"], batched=True)

        # Initialize the model for audio classification based on the number of labels
        num_labels = len(self.id2label)
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )

        # Additional handling for Encodec models if specified
        if 'encodec' in self.model_checkpoint:
            model = EncodecModel.from_pretrained(self.model_checkpoint)

        model_name = self.model_checkpoint.split("/")[-1]

        # Set the output directory based on flags like 'compressed' or 'rerecorded'
        if self.compressed:
            output_dir_name = f"{self.output_dir}/{model_name}-{self.dataset_name}-{self.classification_type}-{str(self.num_epochs)}-finetuned-{str(self.max_duration)}-{str(self.max_size)}-compressed"
        elif self.rerecorded:
            output_dir_name = f"{self.output_dir}/{model_name}-{self.dataset_name}-{self.classification_type}-{str(self.num_epochs)}-finetuned-{str(self.max_duration)}-{str(self.max_size)}-rerecorded"
        else:
            output_dir_name = f"{self.output_dir}/{model_name}-{self.dataset_name}-{self.classification_type}-{str(self.num_epochs)}-finetuned-{str(self.max_duration)}-{str(self.max_size)}"

        # Training arguments for the model
        args = TrainingArguments(
            output_dir_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            warmup_ratio=0.1,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            report_to="none",  # Disable reporting to external services
            max_steps=-1,
        )

        # Initialize the Trainer for model training and evaluation
        trainer = Trainer(
            self.model,
            args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["test"],
            tokenizer=self.feature_extractor,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()
        return

    # Compute evaluation metrics such as accuracy, precision, recall, F1-score
    def compute_metrics(self, eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        if self.classification_type == 'ATTACK':
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        else:
            computed_metrics = {}
            for name, func in self.metrics.items():
                computed_metrics.update(func.compute(predictions=predictions, references=labels))

        return computed_metrics

    # Predict function to make predictions on random samples from the dataset
    def predict(self):
        for i in range(10):
            idx = random.randint(1, self.dataset_size)
            data_item = self.dataset[idx]
            inputs = self.processor(data_item["audio"], sampling_rate=data_item["sr"], return_tensors="pt")
            print("label {}: {}".format(idx, data_item['label']))
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_class_ids = torch.argmax(logits, dim=-1).item()
            print("pred {}: {}".format(idx, predicted_class_ids))
        return

# Dataset class for ASVSpoof dataset
class ASVSpoofDataset(object):
    def __init__(self, sr=16000, root_dir=None, max_size=None, classification_type='KEY', compressed=False, rerecorded=False):

        self.dataset = None
        self.max_size = max_size  # Max number of samples
        self.classification_type = classification_type  # Classification type
        self.sr = sr  # Sampling rate for audio
        self.root_dir = root_dir  # Root directory for dataset

        # Determine the source of the dataset based on flags (compressed or rerecorded)
        if self.root_dir is None:
            if compressed:
                self.root_dir = "azureml://.../compressed_asvspoof5"
                self.fs = AzureMachineLearningFileSystem(self.root_dir)
                metadata_file = "../compressed_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
            elif rerecorded:
                self.root_dir = "azureml://.../rerecorded_flac_D"
                self.fs = AzureMachineLearningFileSystem(self.root_dir)
                metadata_file = "../rerecorded_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/rerecorded_{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
            else:
                self.root_dir = "azureml://.../ASVspoof5"
                self.fs = AzureMachineLearningFileSystem(self.root_dir)
                metadata_file = f"{self.root_dir}/metadata/test_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/flac_D/{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
        else:
            # Handle the dataset if root_dir is provided explicitly
            if compressed:
                metadata_file = f"{self.root_dir}/compressed_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/metadata/{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
            elif rerecorded:
                metadata_file = f"{self.root_dir}/rerecorded_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/rerecorded_flac_D/rerecorded_{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
            else:
                metadata_file = f"{self.root_dir}/metadata/test_metadata.csv"
                metadata = pd.read_csv(metadata_file, nrows=self.max_size)
                self.filenames = [f"{self.root_dir}/flac_D/{file}.flac" for file in metadata['FLAC_FILE_NAME'].to_list()]
                self.labels = metadata[self.classification_type].to_list()
        return

    # Load the dataset and cast columns appropriately (e.g., audio and labels)
    def load_data(self):
        as_dict = {'audio': self.filenames, 'label': self.labels, 'filename': self.filenames}
        self.dataset = Dataset.from_dict(as_dict).cast_column("audio", Audio(sampling_rate=self.sr))
        names = list(set(self.labels))
        ClassLabels = ClassLabel(num_classes=len(names), names=names)
        self.dataset = self.dataset.cast_column('label', ClassLabels)
        print(self.dataset[0])

        return self.dataset

# Dataset class for DeepVoice dataset
class DeepVoiceDataset(object):
    def __init__(self, root_dir, max_size=None, sr=16000, mixed=False):
        self.root_dir = root_dir
        self.dataset = None
        self.max_size = max_size  # Max number of samples
        self.sr = sr  # Sampling rate for audio
        self.mixed = mixed  # Flag for mixed dataset (real and fake)
        return

    # Load the dataset for DeepVoice and handle 'real', 'fake', and optionally 'mixed' data
    def load_data(self):
        real_files = [os.path.join(self.root_dir, 'real', file) for file in os.listdir(os.path.join(self.root_dir, 'real'))]
        fake_files = [os.path.join(self.root_dir, 'fake', file) for file in os.listdir(os.path.join(self.root_dir, 'fake'))]

        if self.max_size is not None:
            random.shuffle(real_files)
            real_files = real_files[:int(self.max_size / 2)] 
            random.shuffle(fake_files)
            fake_files = fake_files[:int(self.max_size / 2)]
        else:
            random.shuffle(fake_files)
            fake_files = fake_files[:len(real_files)]

        self.filenames = real_files + fake_files
        self.labels = ['real'] * len(real_files) + ['spoof'] * len(fake_files)

        if self.mixed:
            mixed_files = [os.path.join(self.root_dir, 'mixed', file) for file in os.listdir(os.path.join(self.root_dir, 'mixed'))]
            self.filenames = self.filenames + mixed_files
            self.labels = self.labels + ['spoof'] * len(mixed_files)

        # Prepare dataset dictionary and cast columns for audio and labels
        as_dict = {'audio': self.filenames, 'label': self.labels, 'filename': self.filenames}
        self.dataset = Dataset.from_dict(as_dict).cast_column("audio", Audio(sampling_rate=self.sr))
        ClassLabels = ClassLabel(num_classes=2, names=['real', 'spoof'])
        self.dataset = self.dataset.cast_column('label', ClassLabels)
        print(self.dataset[0])

        return self.dataset

# Dataset class for FakeAVCeleb dataset
class FakeAVCeleb(object):
    def __init__(self, root_dir, max_size=None, sr=16000):
        self.root_dir = root_dir
        self.max_size = max_size  # Maximum dataset size
        self.sr = sr  # Sampling rate for audio

    # Load data from FakeAVCeleb dataset
    def load_data(self):
        metadata_file = f"{self.root_dir}/FakeAVCeleb-Audio/metadata.csv"
        metadata = pd.read_csv(metadata_file)
        self.filenames = [f"{self.root_dir}/FakeAVCeleb-Audio/{file}" for file in metadata['new_filename'].to_list()]
        self.labels = metadata['category'].to_list()

        as_dict = {'audio': self.filenames, 'label': self.labels, 'filename': self.filenames}
        self.dataset = Dataset.from_dict(as_dict).cast_column("audio", Audio(sampling_rate=self.sr))
        names = list(set(self.labels))
        ClassLabels = ClassLabel(num_classes=len(names), names=names)
        self.dataset = self.dataset.cast_column('label', ClassLabels)
        print(self.dataset[0])

        return self.dataset

# Function to compute Equal Error Rate (EER) for binary classification tasks
def compute_eer(y_true, y_score):
    # Sort by prediction scores
    sorted_indices = np.argsort(y_score)
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]

    # Calculate the False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    far = np.cumsum(y_true == 0) / np.sum(y_true == 0)
    frr = 1 - np.cumsum(y_true == 1) / np.sum(y_true == 1)

    # Find the EER where FAR and FRR intersect
    eer_threshold_index = np.nanargmin(np.abs(far - frr))
    eer = np.mean([far[eer_threshold_index], frr[eer_threshold_index]])

    return eer

# Custom metric for EER calculation in Hugging Face's datasets library
class EERMetric(datasets.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Define information for the EER metric
    def _info(self):
        return datasets.MetricInfo(
            description="Calculate the Equal Error Rate (EER).",
            citation="",
            inputs_description="Binary labels and prediction scores",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("int32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    # Compute the EER metric based on predictions and ground truth labels
    def _compute(self, predictions, references):
        return {
            "eer": compute_eer(np.array(references), np.array(predictions))
        }

# Main function to run the PreTrainedTransformer class
if __name__ == '__main__':
    wav2vec_model = "facebook/wav2vec2-base"
    ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
    encodec_model = "facebook/encodec_24khz"
    
    # Initialize the PreTrainedTransformer with specified arguments from command line
    PreTrainedTransformer(model=args.model_path, max_duration=args.max_duration,
                          num_epochs=args.num_epochs, max_size=args.max_size, 
                          batch_size=args.batch_size, 
                          compressed=False, rerecorded=False,
                          output_dir=args.output_path, dataset_path=args.data,
                          dataset_name=args.dataset_name)
