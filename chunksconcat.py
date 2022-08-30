from dcase_models.data.feature_extractor import FeatureExtractor
import soundfile as sf
import os
import re
import sys
import numpy as np
import json

def main():

    FE = FeatureExtractor()

    #gt_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures2/groundtruthmono'
    #concat_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures2/fullgroundtruth'
    testpath = '/home/christosf/ThesisProject/datasets/MUSDB18-w44100/original/test'
    path = '/home/christosf/ThesisProject/datasets/timestamps/'

    songs = []
    songpaths = []
    for dirpath, dirnames, filenames in os.walk(testpath):
        for dirname in dirnames:
            songs.append(dirname)
            songpaths.append(dirpath + '/' + dirname)

    recsongs = []
    for song in songs:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if song in filename:
                    nsil_path = (dirpath + song + '_nonsilent.json')
                    sil_path = (dirpath + song + '_silences.json')
                    chunks = []
                    data = []
                    with open(nsil_path, 'r') as f:
                        nonsilences = json.load(f)
                    with open(sil_path, 'r') as f2:
                        silences = json.load(f2)
                    ind1 = len(silences)
                    ind2 = len(nonsilences) 
                    for index, item in enumerate(silences):
                        timestamp = item
                        ind = index + 1
                        if ind <= ind1:
                            fn1 = '/home/christosf/ThesisProject/experiments/MUSDB18Mixtures/AttProtos/reconstruction/' + song + '_novocaltrack_' + '{}/'.format(ind) + song + '_novocaltrack_' + '{}'.format(ind) + '.wav'
                            row = [item, fn1]
                            #if fn1 == "PR - Oh No_novocaltrack_15.wav":
                            #    pass
                            #else:
                            chunks.append(fn1)
                        if ind <= ind2:
                            fn2 = '/home/christosf/ThesisProject/experiments/MUSDB18Mixtures/AttProtos/reconstruction/' + song + '_vocaltrack_' + '{}/'.format(ind) + song + '_vocaltrack_' + '{}'.format(ind) + '.wav'
                            #if fn2 == "PR - Oh No_vocaltrack_15.wav":
                            #    pass
                            #else:
                            chunks.append(fn2)
                    recsongs.append(chunks)
                            

    """recsongs = []
                for path in songpaths:
                    for dirpath, dirnames, filenames in os.walk(path):
                        for filename in filenames:
                            if 'vocals.wav' == filename:
                                nsil_path = (dirpath + '/non_silent.json')
                                sil_path = (dirpath + '/silences.json')
                                chunks = []
                                with open(nsil_path, 'r') as f:
                                    nonsilences = json.load(f)
                                with open(sil_path, 'r') as f2:
                                    silences = json.load(f2)
                                    ind1 = len(silences)
                                    ind2 = len(nonsilences) 
                                    for index, item in enumerate(silences):
                                        ind = index + 1
                                        if ind <= ind1:
                                            fn1 = dirpath.split('/')[-1] + '_novocaltrack_' + '{}'.format(ind) + '.wav'
                                            chunks.append(fn1)
                                        if ind <= ind2:
                                            fn2 = dirpath.split('/')[-1] + '_vocaltrack_' + '{}'.format(ind) + '.wav'
                                            chunks.append(fn2)
                                recsongs.append(chunks)"""

    
    """for item in recsongs:
                    s = []
                    file_name = item.split('_')[0]
                    for chunk in item:
                        file = gt_path + '/' + chunk
                        y = FE.load_audio(file, mono=True, change_sampling_rate=False)
                        s.append(y)
                    signal = np.concatenate(s)
                    path = concat_path + '/' + file_name
                    try:
                        os.makedirs(path)
                        print ('Directory ', path, ' Created ')
                    except FileExistsError:
                        print ('Directory ', path, ' already exists')
                    sf.write(path + '/' + file_name, signal, 44100)
            """
    concat_path = '/home/christosf/ThesisProject/experiments/MUSDB18Mixtures/AttProtos/fullreconstruction'

    for item in recsongs:
        s = []
        file_name = item[0].split('/')[-1].split('_')[0]
        print(file_name)
        for chunk in item:
            print(chunk)
            y = FE.load_audio(chunk, mono=True, change_sampling_rate=False)
            s.append(y)
        signal = np.concatenate(s)
        path = concat_path + '/' + file_name
        try:
            os.makedirs(path)
            print ('Directory ', path, ' Created ')
        except FileExistsError:
            print ('Directory ', path, ' already exists')
        sf.write(path + '/' + file_name + '.wav', signal, 44100)
                                


if __name__ == '__main__':
    main()