import os
import sys
import numpy as np
import csv
import mirdata
import logging
import musdb

from dcase_models.util.files import list_wav_files
from dcase_models.data.dataset_base import Dataset

logging.getLogger('sox').setLevel(logging.ERROR)



class MUSDB18Mixtures (Dataset):

    def __init__(self, dataset_path):
            super().__init__(dataset_path)

    def build(self):

        self.audio_path = self.dataset_path
        self.fold_list = ["train", "test"]
        self.label_list = ["vocals", "no_vocals"]
        self.ann_label_list = ["vocals\t\n", "no_vocals\t\n"]
        self.file_lists = {}

    def generate_file_lists(self):
        for fold in self.fold_list:
            audio_folder = os.path.join(self.audio_path, fold)
            self.file_lists[fold] = list_wav_files(audio_folder)

        self.file_to_classes = {}

        for dirpath, foldernames, filenames in os.walk(self.audio_path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    filepath = dirpath + '/' + filename
                    if "_vocals_" in filename:
                        self.file_to_classes[filepath] = 'vocals'
                    if "_novocals_" in filename:
                        self.file_to_classes[filepath] = 'no_vocals'

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        class_name =  self.file_to_classes[file_name]
        class_ix = self.label_list.index(class_name)
        y[:, class_ix] = 1
        return y


class Slakh (Dataset):

    """ Slakh dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for Slakh.

    Url: https://zenodo.org/record/4599666#.YjIiDhPMI0o

    Manilow, E. and Wichern, G. and Seetharaman, P. and Le Roux, J.
    “Cutting Music Source Separation Some {Slakh}: A Dataset to Study 
    the Impact of Training Data Quality and Quantity”, 
    in Proc. IEEE (pp. 559-564), 2019

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/IRMAS).

    Examples
    --------
    To work with IRMAS dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import MedleySolosDb
    >>> dataset = UrbanSound8k('../datasets/MedleySolosDb')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)



    def build(self):
        self.audio_path = self.dataset_path

        self.fold_list = ["train", "test"]

        self.label_list = ["cla", "cel", "gac",
                           "gel", "org", "voi",
                           "flu", "pia", "sax", "tru",
                           "vio"]

        self.ann_label_list = ["cla\t\n", "cel\t\n", "gac\t\n",
                               "gel\t\n", "org\t\n", "voi\t\n",
                               "flu\t\n", "pia\t\n", "sax\t\n", 
                               "tru\t\n", "vio\t\n"]

        self.mirdata = mirdata.initialize('slakh', data_home=self.dataset_path)
        self.data = self.mirdata.load_tracks()
        
        self.file_lists = {}


    def generate_file_lists(self):
        if len(self.file_lists) > 0:
            return True

        fold_tran = {
            "train": "train", "test": "test"}
        self.file_lists = {'train': [], 'test': []}
        
        self.file_to_classes = {}

        for track_id, track_data in self.data.items():
            ann_file = track_data.annotation_path
            self.file_lists[fold_tran[track_data.split]].append(track_data.audio_path)
            if track_data.split == 'train':
                classes = [track_data.predominant_instrument]
                self.file_to_classes[track_data.audio_path] = classes
            if track_data.split == 'test':
                f = open(ann_file, 'r')
                content = f.read()
                data = content.replace('\t', '')
                classes = data.splitlines()
                self.file_to_classes[track_data.audio_path] = classes


    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        for class_name in self.file_to_classes[file_name]:
                class_ix = self.label_list.index(class_name)
                y[:, class_ix] = 1
        return y
            
                
    def download(self, force_download=False):
        self.mirdata.download(cleanup=True, force_overwrite=force_download)
        if self.mirdata.validate():
            self.set_as_downloaded()


class MUSDB18 (Dataset):
    """ MUSDB18 dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for MUSDB18.

    @misc{MUSDB18,
      author       = {Rafii, Zafar and
                      Liutkus, Antoine and
                      Fabian-Robert Stoter and
                      Mimilakis, Stylianos Ioannis and
                      Bittner, Rachel},
      title        = {The {MUSDB18} corpus for music separation},
      month        = dec,
      year         = 2017,
      doi          = {10.5281/zenodo.1117372},
      url          = {https://doi.org/10.5281/zenodo.1117372}
    }


    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../MUSDB18/UrbanSound8k).

    Examples
    --------
    To work with UrbanSound8k dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import UrbanSound8k
    >>> dataset = UrbanSound8k('../datasets/UrbanSound8K')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path + "original")
        self.fold_list = ["train", "test"]
        self.label_list = ["vocals", "bass", "drums", "others"]
        self.file_lists = {}

    def generate_file_lists(self):
        for fold in self.fold_list:
            audio_folder = os.path.join(self.audio_path, fold)
            self.file_lists[fold] = list_wav_files(audio_folder)

        """self.file_to_classes = {}
                        
                                for track_id, track_data in self.data.items():
                                    ann_file = track_data.annotation_path
                                    self.file_lists[fold_tran[track_data.split]].append(track_data.audio_path)
                                    if track_data.split == 'train':
                                        classes = [track_data.predominant_instrument]
                                        self.file_to_classes[track_data.audio_path] = classes
                                    if track_data.split == 'test':
                                        f = open(ann_file, 'r')
                                        content = f.read()
                                        data = content.replace('\t', '')
                                        classes = data.splitlines()
                                        self.file_to_classes[track_data.audio_path] = classes"""

        """self.file_lists = {'train': [], 'test': []}

        mus_train = musdb.DB(root="/home/christosf/AttProtos/datasets/MUSDB18", subsets="train")
        mus_test = musdb.DB(root="/home/christosf/AttProtos/datasets/MUSDB18", subsets="test")
        for item in mus_train:
            self.file_lists['train'].append(item.path)
        for item in mus_test:
            self.file_lists['test'].append(item.path)"""

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = int(os.path.basename(file_name).split('-')[1])
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        zenodo_url = "https://zenodo.org/record/1117372/files"
        zenodo_files = ["musdb18.zip"]
        downloaded = super().download(
            zenodo_url, zenodo_files, force_download
        )
        if downloaded:
            move_all_files_to_parent(self.dataset_path, "MUSDB18")
            self.set_as_downloaded()

class IRMAS (Dataset):
    """ IRMAS dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for IRMAS.

    Url: https://zenodo.org/record/1290750

    Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. 
    “A Comparison of Sound Segregation Techniques for Predominant 
    Instrument Recognition in Musical Audio Signals”, 
    in Proc. ISMIR (pp. 559-564), 2012

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/IRMAS).

    Examples
    --------
    To work with IRMAS dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import MedleySolosDb
    >>> dataset = UrbanSound8k('../datasets/MedleySolosDb')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)



    def build(self):
        self.audio_path = self.dataset_path

        self.fold_list = ["train", "test"]

        self.label_list = ["cla", "cel", "gac",
                           "gel", "org", "voi",
                           "flu", "pia", "sax", "tru",
                           "vio"]

        self.ann_label_list = ["cla\t\n", "cel\t\n", "gac\t\n",
                               "gel\t\n", "org\t\n", "voi\t\n",
                               "flu\t\n", "pia\t\n", "sax\t\n", 
                               "tru\t\n", "vio\t\n"]

        self.mirdata = mirdata.initialize('irmas', data_home=self.dataset_path)
        self.data = self.mirdata.load_tracks()
        
        self.file_lists = {}

    def generate_file_lists(self):
        if len(self.file_lists) > 0:
            return True

        fold_tran = {
            "train": "train", "test": "test"}
        self.file_lists = {'train': [], 'test': []}
        
        self.file_to_classes = {}

        for track_id, track_data in self.data.items():
            ann_file = track_data.annotation_path
            self.file_lists[fold_tran[track_data.split]].append(track_data.audio_path)
            if track_data.split == 'train':
                classes = [track_data.predominant_instrument]
                self.file_to_classes[track_data.audio_path] = classes
            if track_data.split == 'test':
                f = open(ann_file, 'r')
                content = f.read()
                data = content.replace('\t', '')
                classes = data.splitlines()
                self.file_to_classes[track_data.audio_path] = classes


    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        for class_name in self.file_to_classes[file_name]:
                class_ix = self.label_list.index(class_name)
                y[:, class_ix] = 1
        return y
            
                
    def download(self, force_download=False):
        self.mirdata.download(cleanup=True, force_overwrite=force_download)
        if self.mirdata.validate():
            self.set_as_downloaded()

    """def upsample_train_set(self):
        n_instances = np.zeros(len(self.label_list), dtype=int)
        files_by_class = {x: [] for x in range(len(self.label_list))}
        for track_id, track_data in self.data.items():
            if track_data.subset != 'training':
                continue
            n_instances[track_data.instrument_id] += 1
            files_by_class[track_data.instrument_id].append(track_data.audio_path)
        print(n_instances)

        max_instances = np.amax(n_instances)

        print(len(self.file_lists['train']))
        for class_ix in range(len(self.label_list)):
            if n_instances[class_ix] < max_instances:
                new_instances = max_instances - n_instances[class_ix]
                repetitions = int(len(files_by_class[class_ix])/new_instances)
                for j in range(repetitions):
                    self.file_lists['train'].extend(files_by_class[class_ix])
        print(len(self.file_lists['train']))"""


class GoogleSpeechCommands(Dataset):
    """ Google Speech Commands dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for Google Speech Commands.

    Url: https://www.tensorflow.org/datasets/catalog/speech_commands

    Pete Warden
    “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,”
    http://arxiv.org/abs/1804.03209
    August 2018

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Examples
    --------
    To work with GoogleSpeechCommands dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import GoogleSpeechCommands
    >>> dataset = GoogleSpeechCommands('../datasets/GoogleSpeechCommands')


    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = self.dataset_path
        
        self.fold_list = ["train", "validate", "test"]
        self.label_list = ["backward", "bed", "bird",
                           "cat", "dog", "down", "eight",
                           "five", "follow", "forward",
                           "four", "go", "happy",
                           "house", "learn", "left", "marvin",
                           "nine", "no", "off",
                           "on", "one", "right",
                           "seven", "sheila", "six", "stop",
                           "three", "tree", "two",
                           "up", "visual", "wow",
                           "yes", "zero"]

        self.validation_file = os.path.join(self.dataset_path, 'validation_list.txt')
        self.test_file = os.path.join(self.dataset_path, 'testing_list.txt')

    def generate_file_lists(self):
        validation_list = []
        with open(self.validation_file ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')        
            for row in csv_reader:
                validation_list.append(row[0])
                
        test_list = []           
        with open(self.test_file ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')        
            for row in csv_reader:
                test_list.append(row[0])
                
        self.file_lists = {'train': [], 'validate': [], 'test': []}
        all_files = list_wav_files(self.audio_path) 

        for wav_file in all_files:
            path_split = wav_file.split('/')
            base_name = os.path.join(path_split[-2], path_split[-1])
            if base_name in validation_list:
                self.file_lists['validate'].append(wav_file)
                continue
            elif base_name in test_list:
                self.file_lists['test'].append(wav_file)
                continue
            else:
                self.file_lists['train'].append(wav_file)

    def get_annotations(self, file_name, features, time_resolution):
        y = np.zeros((len(features), len(self.label_list)))
        word = file_name.split('/')[-2]
        class_ix = int(self.label_list.index(word))
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        tensorflow_url = "http://download.tensorflow.org/data/"
        tensorflow_files = [
            "speech_commands_v0.02.tar.gz",
            "speech_commands_test_set_v0.02.tar.gz"
        ]
        downloaded = super().download(
            tensorflow_url, tensorflow_files, force_download
        )
        if downloaded:
            self.set_as_downloaded()



class MedleySolosDb(Dataset):
    """ MedleySolosDb dataset.

    This class inherits all functionality from Dataset and
    defines specific attributs and methods for MedleySolosDb.

    Url: https://zenodo.org/record/1344103

    Vincent Lostanlen and Carmine-Emanuele Cella
    “Deep convolutional networks on the pitch spiral for musical instrument recognition,”
    17th International Society for Music Information Retrieval Conference (ISMIR)
    New York, USA, 2016

    Parameters
    ----------
    dataset_path : str
        Path to the dataset fold. This is the path to the folder where the
        complete dataset will be downloaded, decompressed and handled.
        It is expected to use a folder name that represents the dataset
        unambiguously (e.g. ../datasets/UrbanSound8k).

    Examples
    --------
    To work with UrbanSound8k dataset, just initialize this class with the
    path to the dataset.

    >>> from dcase_models.data.datasets import MedleySolosDb
    >>> dataset = UrbanSound8k('../datasets/MedleySolosDb')

    Then, you can download the dataset and change the sampling rate.

    >>> dataset.download()
    >>> dataset.change_sampling_rate(22050)

    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path)

    def build(self):
        self.audio_path = os.path.join(self.dataset_path, 'audio')
        
        self.fold_list = ["train", "validate", "test"] 

        self.label_list = ["clarinet", "distorted electric guitar", "female singer",
                           "flute", "piano", "tenor saxophone", "trumpet",
                           "violin"]
        
        self.mirdata = mirdata.initialize('medley_solos_db', data_home=self.dataset_path)                   
        self.data = self.mirdata.load_tracks()

        self.file_lists = {}

    def generate_file_lists(self):
        if len(self.file_lists) > 0:
            return True

        fold_tran = {
            "training": "train", "validation": "validate", "test": "test"}
        self.file_lists = {'train': [], 'validate': [], 'test': []}
        self.file_to_class = {}
        
        for track_id, track_data in self.data.items():  
            print('track_data:', track_data, 'track_data.subset:', track_data.subset)
            self.file_lists[fold_tran[track_data.subset]].append(track_data.audio_path)
            self.file_to_class[track_data.audio_path] = track_data.instrument_id

    def get_annotations(self, file_name, features, time_resolution):
        print(file_name)
        y = np.zeros((len(features), len(self.label_list)))
        class_ix = self.file_to_class[file_name]
        print(class_ix)
        y[:, class_ix] = 1
        return y

    def download(self, force_download=False):
        self.mirdata.download(cleanup=True, force_overwrite=force_download)
        if self.mirdata.validate():
            self.set_as_downloaded()

    def upsample_train_set(self):
        n_instances = np.zeros(len(self.label_list), dtype=int)
        files_by_class = {x: [] for x in range(len(self.label_list))}
        for track_id, track_data in self.data.items():
            if track_data.subset != 'training':
                continue
            n_instances[track_data.instrument_id] += 1
            files_by_class[track_data.instrument_id].append(track_data.audio_path)
        print(n_instances)

        max_instances = np.amax(n_instances)

        print(len(self.file_lists['train']))
        for class_ix in range(len(self.label_list)):
            if n_instances[class_ix] < max_instances:
                new_instances = max_instances - n_instances[class_ix]
                repetitions = int(len(files_by_class[class_ix])/new_instances)
                for j in range(repetitions):
                    self.file_lists['train'].extend(files_by_class[class_ix])
        print(len(self.file_lists['train']))
