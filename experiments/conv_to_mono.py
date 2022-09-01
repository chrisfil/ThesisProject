from dcase_models.data.feature_extractor import FeatureExtractor
import soundfile as sf
import os
import sys
import numpy as np


def main():

    FE = FeatureExtractor()
    gt_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/test'
    outputdir = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/testmixturesmono'

    for (dirpath, dirnames, filenames) in os.walk(gt_path):
        for file_name in filenames:
            file = dirpath + '/' + file_name
            y = FE.load_audio(file, mono=True, change_sampling_rate=False)
            path = outputdir + '/' + file_name.split('.wav')[0].replace('vocals', 'vocaltrack')
            try:
                os.makedirs(path)
                print ('Directory ', path, ' Created ')
            except FileExistsError:
                print ('Directory ', path, ' already exists')

            sf.write(path + '/' + file_name.replace('vocals', 'vocaltrack'), y, 44100)


if __name__ == '__main__':
    main()