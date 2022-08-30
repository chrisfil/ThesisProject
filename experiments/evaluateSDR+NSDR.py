import os
import argparse
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

from tensorflow.keras.layers import BatchNormalization
import keras
import json
import scipy
import scipy.signal
import scipy.io.wavfile
import math
import time
import wave
from pylab import *
import array
from os.path import expanduser
from IPython.display import Audio, display
import soundfile as sf

sys.path.append('../')

# Datasets
from apnet.datasets import MedleySolosDb, IRMAS, MUSDB18Mixtures

# Models
from attprotos.model import AttProtos, AttProtos2

# Features
from dcase_models.data.features import MelSpectrogram, STFT_Phase
from attprotos.features import Openl3
from attprotos.losses import prototype_loss, dummy_loss
from attprotos.layers import Layer, PrototypeReconstruction
from dcase_models.data.data_generator import DataGenerator
from dcase_models.util.files import load_json, load_pickle, save_pickle, mkdir_if_not_exists
from dcase_models.data.scaler import Scaler
from dcase_models.util.data import evaluation_setup
from dcase_models.model.container import KerasModelContainer
from dcase_models.data.feature_extractor import FeatureExtractor

import matplotlib.pyplot as plt

from keras.layers import Input, Lambda, Dense, Flatten, Multiply, Reshape, Concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import LeakyReLU, Activation, ReLU
from keras.models import Model
from keras.regularizers import l1
import keras.backend as K

import tensorflow as tf

import librosa
from librosa.feature.inverse import mel_to_stft
from librosa.util import nnls
import librosa.display

import essentia
import essentia.standard
from essentia.standard import *

#rootdir_path = '/home/christosf/AttProtos/xaudio-main'
#sys.path.append(rootdir_path)
#from xaudio.core import create_analyzer
#from xaudio.models import yamnet, vggish, attprotos
#import innvestigate.utils as iutils
#import innvestigate

import shutil
import museval
import os.path as op
import glob
import functools
import musdb
import warnings
import pandas as pd

rootdir_path = '/home/christosf/anaconda3/envs/apnet2/lib/python3.6/site-packages/museval'
sys.path.append(rootdir_path)
import metrics

from aggregate import TrackStore, MethodStore, EvalStore, json2df


available_models = {
    'AttProtos' : AttProtos,
    'AttProtos2' : AttProtos2
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
    'Openl3' : Openl3
}

available_datasets = {
    'MedleySolosDb' :  MedleySolosDb,
    'IRMAS' : IRMAS,
    'MUSDB18Mixtures' : MUSDB18Mixtures
}


def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))


def eval_dir(
    reference_dir,
    estimates_dir,
    output_dir=None,
    mode='v4',
    win=1.0,
    hop=1.0,
):
    """Compute bss_eval metrics for two given directories assuming file
    names are identical for both, reference source and estimates.
    Parameters
    ----------
    reference_dir : str
        path to reference sources directory.
    estimates_dir : str
        path to estimates directory.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in
    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """

    reference = []
    estimates = []

    data = TrackStore(
        win=win, hop=hop,
        track_name=os.path.basename(reference_dir)
    )

    silent_tracks = []

    global_rate = None
    reference_glob = os.path.join(reference_dir, '*.wav')
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(reference_glob):
        ref_audio, rate = sf.read(
            reference_file,
            always_2d=True
        )
        # Make sure fs is the same for all files
        assert (global_rate is None or rate == global_rate)
        global_rate = rate
        #ref = librosa.to_mono(y.T)
        #ref_audio = np.expand_dims(ref, axis=1)
        ref_audio = np.expand_dims(ref_audio, axis=0)
        if _any_source_silent(ref_audio):
            print("Reference source is silent: ", reference_dir)
            silent_tracks.append(reference_dir)
        #else:
        #    silent_tracks.append(reference_dir, "False")
        reference.append(ref_audio)

    if not reference:
        raise ValueError('`reference_dir` contains no wav files')

    estimated_glob = os.path.join(estimates_dir, '*.wav')
    targets = []
    for estimated_file in glob.glob(estimated_glob):
        targets.append(os.path.basename(estimated_file))
        est_audio, rate = sf.read(
            estimated_file,
            always_2d=True
        )
        assert (global_rate is None or rate == global_rate)
        global_rate = rate
        est_audio = np.expand_dims(est_audio, axis=0)
        if _any_source_silent(est_audio):
            print("Reference source is silent: ", estimates_dir)
            silent_tracks.append(estimates_dir)
        #else:
        #    silent_tracks.append(estimates_dir, "False")
        estimates.append(est_audio)

    for i, target in enumerate(targets):
        est = np.array(estimates[i])
        ref = np.array(reference[i])

        #t, t2 = pad_or_truncate(ref, est)
        t = ref
        t2 = est

        SDR, ISR, SIR, SAR, _ = metrics.bss_eval(
            t,
            t2,
            compute_permutation=True,
            window=win*44100,
            hop=hop*44100,
            framewise_filters=(mode),
            bsseval_sources_version=False
        )
        
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist()
        }

        data.add_target(
            target_name=target,
            values=values
        )

        print(data)
        return data, values["SDR"], silent_tracks


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='UrbanSound8k'
    )
    parser.add_argument(
        '-f', '--features', type=str,
        help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
        default='MelSpectrogram'
    )
    parser.add_argument(
        '-m', '--model', type=str,
        help='model name (e.g. APNet, MLP, SB_CNN ...)')
    
    parser.add_argument('-fold', '--fold_name', type=str, help='fold name')

    parser.add_argument(
        '-mp', '--models_path', type=str,
        help='path to load the trained model',
        default='./'
    )
    parser.add_argument(
        '-dp', '--dataset_path', type=str,
        help='path to load the dataset',
        default='./'
    )
    parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                                    default='0')

    parser.add_argument('-full', '--fullsong', type=bool, help='fullsong',
                                    default=False)

    args = parser.parse_args()

    # only use one GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible
    #print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    if args.features not in available_features:
        raise AttributeError('Features not available')

    model_name = args.model
    if args.model not in available_models:
        base_model = args.model.split('/')[0]
        if base_model not in available_models:
            raise AttributeError('Model not available')
        else:
            model_name = base_model
                
    fullsong = args.fullsong
    # Model paths
    model_folder = os.path.join(args.models_path, args.dataset, args.model)

    # Get parameters
    parameters_file = os.path.join(model_folder, 'config.json')
    params = load_json(parameters_file)

    params_features = params['features'][args.features]
    if 'pad_mode' in params_features:
        if params_features['pad_mode'] == 'none':
            params_features['pad_mode'] = None
    params_dataset = params['datasets'][args.dataset]
    params_model = params['models'][model_name]

    # Get and init dataset class
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path)

    dataset.generate_file_lists()
            
    vocals_train = []
    no_vocals_train = []
    for item in dataset.file_lists['train']:
        class_ = dataset.file_to_classes[item]
        if class_ == 'vocals':
            vocals_train.append(item)
        if class_ == 'no_vocals':
            no_vocals_train.append(item)
            
    vocals_test = []
    no_vocals_test = []
    for item in dataset.file_lists['test']:
        class_ = dataset.file_to_classes[item]
        if class_ == 'vocals':
            vocals_test.append(item)
        if class_ == 'no_vocals':
            no_vocals_test.append(item)
                        
            
    """vocals_val = []
                no_vocals_val = []
                for item in dataset.file_lists['validate']:
                    class_ = dataset.file_to_classes[item]
                    if class_ == 'vocals':
                        vocals_val.append(item)
                    if class_ == 'no_vocals':
                        no_vocals_val.append(item)"""

    print(dataset.label_list)
    print("train set:", len(dataset.file_lists['train']), "tracks in total", "(vocals:", len(vocals_train), ", ", "no_vocals:", len(no_vocals_train), ")" )
    print("test set:", len(dataset.file_lists['test']), "tracks in total", "(vocals:", len(vocals_test), ", ", "no_vocals:", len(no_vocals_test), ")" )
    #print("validate set:", len(dataset.file_lists['validate']), "tracks in total", "(vocals:", len(vocals_val), ", ", "no_vocals:", len(no_vocals_val), ")" )


    dataset_name = args.dataset
    models_path = '/home/christosf/AttProtos/experiments'
    fold_name = "test"


    model_folder = os.path.join(models_path, dataset_name, args.model)
    parameters_file = os.path.join(model_folder, 'config.json')

    with open(parameters_file, 'r') as f:
        params = json.load(f)

    params_features = params['features']["MelSpectrogram"]

    params_dataset = params['datasets'][dataset_name]
    params_model = params['models'][args.model]
            

    # Get and init dataset class

    #dataset_class = available_datasets[args.dataset]
    #dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
    #dataset = dataset_class(dataset_path)

    #dataset.generate_file_lists()

    # Get and init feature class
    features = MelSpectrogram(**params_features)


    scaler = Scaler(normalizer=params_model['normalizer'])
    # Init data generator
    data_gen_test = DataGenerator(
        dataset, features, folds=[fold_name],
        batch_size=params['train']['batch_size'],
        shuffle=False, train=False, scaler=scaler
    )


    metrics = ['classification']

    features_shape = features.get_shape()
    n_frames_cnn = features_shape[1]
    n_freq_cnn = features_shape[2]
    n_classes = len(dataset.label_list)

    print(dataset.label_list)
    print(n_classes)
    print(n_freq_cnn)
    print(n_frames_cnn)
    print('Features shape: ', features.get_shape(10.0))

    model_class = available_models[args.model]

    model_container = model_class(
        model=None, model_path=None, n_classes=n_classes,
        n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
        metrics=metrics, training=True,
        **params_model['model_arguments']
    )

    exp_folder = os.path.join(models_path, dataset_name, args.model, args.fold_name)


    scaler_file = os.path.join(exp_folder, 'scaler.pickle')
    scaler = load_pickle(scaler_file)

    data_gen_test = DataGenerator(
        dataset, features, folds=[fold_name],
        batch_size=params['train']['batch_size'],
        shuffle=False, train=False, scaler=scaler
    )

    data_gen_train = DataGenerator(
        dataset, features, folds=['train'],
        batch_size=params['train']['batch_size'],
        shuffle=False, train=True, scaler=scaler
    )

    model = model_container.model

    weights_path = os.path.join(exp_folder, 'best_weights.hdf5')

    model.load_weights(weights_path)


    X, Y = data_gen_test.get_data()
    print(model.outputs)

    
    if fullsong:
        rec_path = os.path.join(model_folder, 'fullreconstruction')
        print('Rec Path', rec_path)
        gtv_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/fullgroundtruth'
        gtx_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/reconstructedmixtures'

        evaluationx = []
        evaluationv = []
        data = []

        i=0
        gtx_file_names = []
        for dirname, fn, files in os.walk(gtx_path):
            if i is not 0:
                gtx_file_names.append(dirname)
            i = i+1

        i=0
        gtv_file_names = []
        for dirname, fn, files in os.walk(gtv_path):
            if i is not 0:
                gtv_file_names.append(dirname)
            i = i+1

        i=0
        estfile_names = []
        for dirname, fn, files in os.walk(rec_path):
            if i is not 0:
                estfile_names.append(dirname)
            i = i+1


        for i in range(len(estfile_names)):
            data_name = estfile_names[i].split('/')[-1]
            print(i)

            print(gtv_file_names[i])

            print(gtx_file_names[i])
            
            print(estfile_names[i])
            
            file_gtv = gtv_file_names[i] + '/' + data_name + '.wav'
            dur = librosa.get_duration(filename=file_gtv)
            print(dur)


            resultsx = museval.eval_dir(
                gtv_file_names[i],
                gtx_file_names[i],
                output_dir='/home/christosf/AttProtos/datasets/MUSDB18Mixtures/musevalout',
                mode='v4',
                win=1.0,
                hop=1.0
            )
            evaluationx.append(resultsx)

            resultsv = museval.eval_dir(
                gtv_file_names[i],
                estfile_names[i],
                output_dir='/home/christosf/AttProtos/datasets/MUSDB18Mixtures/musevalout',
                mode='v4',
                win=1.0,
                hop=1.0
            )
            evaluationv.append(resultsv)

            #SDRx = float(str(evaluationx[i]).split('SDR: ')[1].split(' SIR:')[0])

            SDRx = float(str(evaluationx[i]).split("==> SDR: ")[1].split("  SIR:")[0])
            #print('SDRx: ', SDRx)

            SDRv = float(str(evaluationv[i]).split("==> SDR: ")[1].split("  SIR:")[0])
            print('SDRv: ', SDRv)

            NSDR = SDRv - SDRx
            print('NSDR: ', NSDR)

            row = [data_name, dur, resultsv, SDRx, SDRv, NSDR]

            data.append(row)

            header = ['file_name', 'duration', 'eval_results', 'SDR_gtmixture', 'SDRvocals', 'NSDR']
            with open('fullsongSDR+NSDR{}.csv'.format('_' + args.dataset + '_' + args.model + '_' + args.fold_name), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)

        df = pd.read_csv('fullsongSDR+NSDR{}.csv'.format('_' + args.dataset + '_' + args.model + '_' + args.fold_name))

        df1 = df['SDRvocals']
        df2 = df['NSDR']

        valuesSDR = []
        for value in df1:
            #values.append(float(value.replace('[', '').replace(']', '')))
            valuesSDR.append(value)

        valuesNSDR = []
        for value in df2:
            #values.append(float(value.replace('[', '').replace(']', '')))
            valuesNSDR.append(value)

         
        # Creating valuesSDR plot
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(valuesSDR)
        plt.title('SDR Boxplot')
        plt.ylabel('SDR (dB)')
        plt.show()
        plt.savefig('boxplot_fullsongSDR{}.png'.format('_' + args.dataset + '_' + args.model + '_' + args.fold_name))

        # Creating NSDR plot
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(valuesNSDR)
        plt.title('NSDR Boxplot')
        plt.ylabel('NSDR (dB)')
        plt.show()
        plt.savefig('boxplot_fullsongNSDR{}.png'.format('_' + args.dataset + '_' + args.model + '_' + args.fold_name))

    else:

        rec_path = os.path.join(model_folder, 'reconstruction2withwin')
        print('Rec Path', rec_path)
        gtv_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/groundtruthmono'
        gtx_path = '/home/christosf/AttProtos/datasets/MUSDB18Mixtures/testmixturesmono'

        evaluationx = []
        evaluationv = []
        data = []

        df = pd.read_csv('/home/christosf/AttProtos/experiments/silent_tracks.csv')
        sts = []
        for index, row in enumerate(df['silent_tracks']):
            if row != '[]':
                chunk = row.split('/')[-1].replace("']","")
                sts.append(chunk)

        i=0
        gtx_file_names = []
        for dirname, fn, files in os.walk(gtx_path):
            if i is not 0:
                if dirname.split('/')[-1] not in sts:
                    gtx_file_names.append(dirname)
            i = i+1

        i=0
        gtv_file_names = []
        for dirname, fn, files in os.walk(gtv_path):
            if i is not 0:
                if dirname.split('/')[-1] not in sts:
                    gtv_file_names.append(dirname)
            i = i+1

        i=0
        estfile_names = []
        for dirname, fn, files in os.walk(rec_path):
            if i is not 0:
                if dirname.split('/')[-1] not in sts:
                    estfile_names.append(dirname)
            i = i+1

        data = []
        data2 = []
        evaluationv = []
        evaluationx = []
        for i in range(1560, len(X)):
            #for i in range(900):
            print(i)
            data_name = dataset.file_lists['test'][i].split('/')[-1]
            if data_name.split('.wav')[0].replace('vocals', 'vocaltrack') not in sts:
                print('Rec Path', rec_path)
                print(data_name)

                mel_spec_data = np.concatenate(X[i], axis=0)

                class_gt = dataset.file_to_classes[dataset.file_lists['test'][i]]
                print(class_gt)
                
                predictions = list(model_container.model.predict(X[i])[0][0])
                
                pred = np.max(predictions)
                print(pred)
                
                class_most_activated = predictions.index(pred)
                print(class_most_activated)
                
                pred_class = dataset.label_list[class_most_activated]
                print(pred_class)
                
                #if class_gt == 'vocals':
                t = gtv_path + '/' + data_name.split('.wav')[0].replace('vocals', 'vocaltrack')
                #print(t)
                t2 = rec_path + '/' + data_name.split('.wav')[0].replace('vocals', 'vocaltrack')

                file_est = t2 + '/' + data_name.replace('vocals', 'vocaltrack')
                dur = librosa.get_duration(filename=file_est)
                print(dur)

                resultsv = eval_dir(
                    t,
                    t2,
                    output_dir='/home/christosf/AttProtos/datasets//musevalout',
                    mode='v4',
                    win=1.0,
                    hop=1.0,
                )

                evaluationv.append(resultsv)
                #print(resultsv)

                t4 = gtx_path + '/' + data_name.split('.wav')[0].replace('vocals', 'vocaltrack')
                print(t4)
                resultsx = eval_dir(
                    t,
                    t4,
                    output_dir='/home/christosf/AttProtos/datasets/MUSDB18Mixtures/musevalout',
                    mode='v4',
                    win=1.0,
                    hop=1.0,
                )
                evaluationx.append(resultsx)

                SDRx = float(str(resultsx[0]).split("==> SDR: ")[1].split("  SIR:")[0])
                #print(resultsx)
                
                SDRv = float(str(resultsv[0]).split("==> SDR: ")[1].split("  SIR:")[0])
                print('SDRv: ', SDRv)

                NSDR = SDRv - SDRx
                print('NSDR: ', NSDR)

                SDR_values = resultsv[1]

                MEAN_SDR = mean(resultsv[1])

                print(SDRv)
                print(MEAN_SDR)

                row = [data_name, dur, mel_spec_data, predictions, class_gt, pred_class, resultsv[0], resultsv[1], SDRx, SDRv, MEAN_SDR, NSDR]
                #row = [eval_results[0], SDR[0]]

                #row2 = [resultsv[2]]

                data.append(row)
                #data2.append(row2)
                header = ['file_name', 'duration', 'mel_spec', 'predictions', 'ground_truth', 'predicted_class', 'eval_results', 'eval_results_per_window', 'SDR_gtmixture', 'SDRvocals', "mean_SDR", 'NSDR']
                #header = ['eval_results', 'SDR']

                #header2 = ['silent_tracks']

                with open('chunksSDR+NSDR+classpredictions2{}.csv'.format('_' + args.dataset + '_' + args.model +'_' + args.fold_name), 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data)

                """with open('silent_tracks_with_voc.csv', 'w', encoding='UTF8', newline='') as f2:
                                                        writer = csv.writer(f2)
                                                        writer.writerow(header2)
                                                        writer.writerows(data2)"""

        """# Plot SDR boxplot:
                        
                                df = pd.read_csv('chunks_voc_SDR+NSDR+classpredictions{}.csv'.format('_' + args.dataset + '_' + args.model +'_' + args.fold_name))
                        
                                df1 = df['SDR']
                                values = []
                                for value in df1:
                                    #values.append(float(value.replace('[', '').replace(']', '')))
                                    values.append(value)
                        
                                fig = plt.figure(figsize =(10, 7))
                                 
                                # Creating plot
                                plt.boxplot(values)
                                plt.title('SDR Boxplot')
                                plt.ylabel('SDR (dB)')
                        
                                # show plot
                                plt.show()
                                plt.savefig('boxplot_chunksSDR{}.png'.format('_' + args.dataset + '_' + args.fold_name + '-lrp_alpha1beta0_mono'))
                                    #plt.savefig('boxplot{}.png')"""

        

if __name__ == "__main__":
    main()
