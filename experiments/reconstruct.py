from tensorflow.keras.layers import BatchNormalization
import keras
import os
import sys
import numpy as np
import json
import argparse
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


#rootdir_path = '/home/christosf/AttProtos/apnet'
#sys.path.append(rootdir_path)
sys.path.append('../')
from apnet.datasets import MUSDB18Mixtures

#rootdir_path = '/home/christosf/AttProtos'
#sys.path.append(rootdir_path)

from attprotos.layers import PrototypeReconstruction
from attprotos.losses import prototype_loss
from attprotos.model import AttProtos, AttProtos2
from attprotos.features import Openl3

# Features
from dcase_models.data.features import MelSpectrogram, STFT_Phase

from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json, load_pickle
from dcase_models.util.files import mkdir_if_not_exists, save_pickle
from dcase_models.util.data import evaluation_setup
from dcase_models.model.container import KerasModelContainer
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.util.files import load_json

from keras.layers import Layer

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.layers import Input, Lambda, Dense, Flatten, Multiply, Reshape, Concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, AveragePooling2D
from keras.layers import LeakyReLU, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l1
import keras.backend as K

from dcase_models.model.container import KerasModelContainer
#rootdir_path = '/home/christosf/AttProtos'
#sys.path.append(rootdir_path)
from attprotos.losses import prototype_loss, dummy_loss
from attprotos.layers import PrototypeReconstruction

import tensorflow as tf

rootdir_path = '/home/christosf/AttProtos/xaudio-main'
sys.path.append(rootdir_path)
from xaudio.core import create_analyzer
from xaudio.models import yamnet, vggish, attprotos
import innvestigate.utils as iutils
import matplotlib.pyplot as plt
import innvestigate

import librosa
from librosa.feature.inverse import mel_to_stft
from librosa.util import nnls
import librosa.display

import soundfile as sf

import essentia
import essentia.standard
from essentia.standard import *

rootdir_path = '/home/christosf/softmaxgradient-lrp'
sys.path.append(rootdir_path)
from utils.visualizations import SGLRP
import innvestigate.utils as iutils


available_models = {
    'AttProtos' : AttProtos,
    'AttProtos2' : AttProtos2
}

available_datasets = {
    'MUSDB18Mixtures' : MUSDB18Mixtures
}


class Analyzer:
    def __init__(self, analyzer, model, neuron_selection_mode):
        self.analyzer = innvestigate.create_analyzer(
            analyzer, model, neuron_selection_mode=neuron_selection_mode
        )

    def analyze(self, filename, index=None, **kwargs):
        inputs = filename
        if index is not None:
            analysis = self.analyzer.analyze(inputs, index, **kwargs)
        else:
            analysis = self.analyzer.analyze(inputs, **kwargs)
        return inputs, analysis


def create_analyzer(
    analyzer, model, neuron_selection_mode="max_activation"
):
    analysis = Analyzer(analyzer, model, neuron_selection_mode)
    return analysis


def overlapadd(speech, time_context, overlap, y):
    sep1 = np.zeros((((len(speech)*(time_context-overlap)+time_context), )))
    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    i=0
    start=0

    for sig in speech:
        s1=sig
        if start==0:
            sep1[0:time_context] = s1
            sep1[0:overlap] = window[overlap:]*sep1[start:start+overlap] + window[:overlap]*s1[:overlap]
        if i == len(speech)-1:
            sep1[start+overlap:start+time_context] = s1[overlap:time_context]
            sep1[start:start+overlap] = window[overlap:]*sep1[start:start+overlap] + window[:overlap]*s1[:overlap]
            sep=sep1[0:len(y)]
        else:
            sep1[start+overlap:start+time_context] = s1[overlap:time_context]
            sep1[start:start+overlap] = window[overlap:]*sep1[start:start+overlap] + window[:overlap]*s1[:overlap]
        i = i + 1 #index for each block
        start = start - overlap + time_context
    return sep

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
        '-m', '--model', type=str,
        help='model name (e.g. APNet, MLP, SB_CNN ...)')
    
    parser.add_argument('-fold', '--fold_name', type=str, help='fold name')

    
    parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                        default='0')

    args = parser.parse_args()

    # only use one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible
    print(__doc__)

    dataset_name = args.dataset
    models_path = '/home/christosf/AttProtos/experiments'
    dataset_path = './'
    fold_name = "test"


    model_folder = os.path.join(models_path, dataset_name, args.model)
    parameters_file = os.path.join(model_folder, 'config.json')

    with open(parameters_file, 'r') as f:
        params = json.load(f)

    params_features = params['features']["MelSpectrogram"]

    params_dataset = params['datasets'][dataset_name]
    params_model = params['models'][args.model]

    dataset_path = params_dataset['dataset_path']
    dataset_class = available_datasets[args.dataset]
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

    print("model:", args.model)
    print("weights fold:", args.fold_name)
    print("labels:", dataset.label_list)
    print("train set:", len(dataset.file_lists['train']), "tracks in total", "(vocals:", len(vocals_train), ", ", "no_vocals:", len(no_vocals_train), ")" )
    print("test set:", len(dataset.file_lists['test']), "tracks in total", "(vocals:", len(vocals_test), ", ", "no_vocals:", len(no_vocals_test), ")" )
    #print("validate set:", len(dataset.file_lists['validate']), "tracks in total", "(vocals:", len(vocals_val), ", ", "no_vocals:", len(no_vocals_val), ")" )




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

    #print('Labels: ', dataset.label_list)
    #print('Number of classes: ',n_classes)
    #print('n_freq_cnn: ', n_freq_cnn)
    #print('n_frames_cnn: ', n_frames_cnn)
    #print('Features shape: ', features.get_shape(10.0))
    model_class = available_models[args.model]

    model_container = model_class(
        model=None, model_path=None, n_classes=n_classes,
        n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
        metrics=metrics, training=True,
        **params_model['model_arguments']
    )


    exp_folder = os.path.join('/home/christosf/AttProtos/experiments/', args.dataset, args.model, args.fold_name)
    model = model_container.model

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
    print(weights_path)

    model.load_weights(weights_path)

    X, Y = data_gen_test.get_data()
    print('Outputs: ', model.outputs)
    
        
    print("Reconstructing:")

    if args.model == "AttProtos":
        FE = FeatureExtractor()
        instance = 0
        for file in dataset.file_lists['test']:
            if dataset.file_to_classes[file] == 'vocals':
                file_name = file.split('/')[-1].replace('vocals', 'vocaltrack')
            elif dataset.file_to_classes[file] == 'no_vocals':
                file_name = file.split('/')[-1].replace('novocals', 'novocaltrack')
            
            outputdir = os.path.join('/home/christosf/AttProtos/experiments/', args.dataset, args.model, 'reconstruction')
            
            print(instance, file_name)
            
            y = FE.load_audio(file, mono=True, change_sampling_rate=False)
            
            features = STFT_Phase(**params_features, full_spectrogram=False)
            phase = features.calculate(file)
                
            time_context = features.sequence_frames
            overlap = int((features.sequence_frames/features.sequence_time)*2)
            #spec_v = []
            #spec_a = []
            speech= []
            #accomp = []
            for i in range(len(X[instance])):
                class_ix = 0
                pred = model_container.model.predict(X[instance])[0][i].reshape(1,-1)
                
                logits_voc = np.zeros_like(pred)
                logits_voc[0][0] = pred[0][0]
                
                logits_acc = np.zeros_like(pred)
                logits_acc[0][1] = pred[0][1]
                
                alpha = model_container.model_encoder_mask.predict(X[instance])[i]
                #print(alpha.shape)

                w = model_container.model.get_layer('dense').get_weights()[0]
                #print(w.shape)
                grad_voc = logits_voc.dot(w.T)
                grad_voc = np.reshape(grad_voc, (1, 64, 15))
                
                grad_acc = logits_acc.dot(w.T)
                grad_acc = np.reshape(grad_acc, (1, 64, 15))
                
                alpha2 = np.sum(alpha, axis=0)
                #print(alpha2.shape)
                
                alpha_grad_voc = grad_voc*alpha2
                alpha_grad_voc = alpha_grad_voc*(alpha_grad_voc>0)
                
                alpha_grad_acc = grad_acc*alpha2
                alpha_grad_acc = alpha_grad_acc*(alpha_grad_acc>0)
                
                data = X[instance][i]
                
                scaled = X[instance][i]
                #scaled = frame
                inversed = scaler.inverse_transform(scaled)
                
                melspec = librosa.db_to_power(inversed)

                saliency_voc = np.zeros_like(data)
                
                saliency_acc = np.zeros_like(data)
                #mask_i = saliency[true_class] / sum(saliency)
                #melspec_i = mask_i * data

                energy_voc = np.sum(alpha_grad_voc[0]**2, axis=0) # (15,)
                energy_acc = np.sum(alpha_grad_acc[0]**2, axis=0) # (15,)
                
                profile_voc = alpha_grad_voc[0, :, np.argmax(energy_voc)] # (64,)
                profile_acc = alpha_grad_acc[0, :, np.argmax(energy_acc)] # (64,)

                profile_voc_extend = np.interp(np.arange(256), np.arange(64)*4, profile_voc)
                profile_voc_extend = np.convolve(profile_voc_extend, [1/64]*64, mode='same') # (128,)
                saliency_voc = profile_voc_extend.copy() 
                #print(saliency_voc.shape)
                
                profile_acc_extend = np.interp(np.arange(256), np.arange(64)*4, profile_acc)
                profile_acc_extend = np.convolve(profile_acc_extend, [1/64]*64, mode='same') # (128,)
                saliency_acc = profile_acc_extend.copy() 
                #print(saliency_acc.shape)
                
                #masked_voc_data = 2*((data+1)*saliency/2) - 1 # (128x256)
                #print(masked_data.shape)
                mask_0 = saliency_voc
                #mask_0 = np.expand_dims(mask_0, 0)
                
                mask_1 = saliency_acc
                #mask_1 = 1 - mask_0
                #mask_1 = np.expand_dims(mask_1, 0)
                
                mask_voc = mask_0/(mask_0+mask_1)
                
                #mask_acc = mask_1/(mask_0+mask_1)
                
                mask_voc = scipy.special.expit(mask_voc)
                mask_voc = np.log10(0.01+mask_voc)
                mask_voc = (mask_voc - mask_voc.min()) / (mask_voc.max() - mask_voc.min())
                
                #mask_acc = 1 - mask_voc 
                
                
                #mask_acc = scipy.special.expit(mask_1)
                #mask_acc = np.log10(0.01+mask_acc)
                #mask_acc = (mask_acc - mask_acc.min()) / (mask_acc.max() - mask_acc.min())
                
                masked_spec_voc = melspec*mask_voc
                #masked_data_voc = mask_0*data
                #masked_spec_acc = melspec*mask_acc
                
                inverse_voc = nnls(features.mel_basis, masked_spec_voc.T)
                inverse_voc = np.power(inverse_voc, 1./2.0, out=inverse_voc)
                
                #inverse_acc = nnls(features.mel_basis, masked_spec_acc.T)
                #inverse_acc = np.power(inverse_acc, 1./2.0, out=inverse_acc)
                
                polar_speech = inverse_voc * np.cos(phase[i].T) + inverse_voc * np.sin(phase[i].T) * 1j
                speech_out = librosa.istft(polar_speech, hop_length=features.audio_hop, win_length=features.audio_win, n_fft=features.n_fft, window='hann', center=True)
                
                #polar_acc = inverse_acc * np.cos(phase[i].T) + inverse_acc * np.sin(phase[i].T) * 1j
                #acc_out = librosa.istft(polar_acc, hop_length=features.audio_hop, win_length=features.audio_win, n_fft=features.n_fft, window='hann', center=True)

                #masked_data_acc = 1-masked_data_voc
                #plt.imshow(masked_data_voc.T, origin='lower')
                #plt.colorbar()
                #images.append(np.expand_dims(masked_data, 0))
                speech.append(speech_out)
                #accomp.append(acc_out)
                
                #spec_v.append(masked_spec_voc)
                #spec_a.append(masked_spec_acc)
                
            time_context = len(speech[0])
            #overlap = int((time_context/(features.sequence_frames/features.sequence_hop))*2)
            #overlap = features.audio_hop*(int((features.sequence_hop_time/features.sequence_time)*features.sequence_frames))
            overlap = int(time_context/2) - int(features.audio_hop/2)

            signal = overlapadd(speech, time_context, overlap, y)

            path = outputdir + '/' + file_name.split('.wav')[0]

            try:
                os.makedirs(path)    
                print("Directory " , path ,  " Created ")
            except FileExistsError:
                print("Directory " , path ,  " already exists") 

            sf.write(path + '/' + file_name, signal, 44100)

            #sf.write(outputdir + '/' + file_name, signal, 44100)
                
            instance = instance +1

    elif args.model == "AttProtos2":
        FE = FeatureExtractor()
        instance = 0
        for file in dataset.file_lists['test']:
            if dataset.file_to_classes[file] == 'vocals':
                file_name = file.split('/')[-1].replace('vocals', 'vocaltrack')
            if dataset.file_to_classes[file] == 'no_vocals':
                file_name = file.split('/')[-1].replace('no_vocals', 'novocaltrack')
                #print(instance)
            print(instance, file_name)
            y = FE.load_audio(file, mono=True, change_sampling_rate=False)
            file_name = file.split('/')[-1].replace('vocals', 'vocaltrack')
            features = STFT_Phase(**params_features, full_spectrogram=False)
            phase = features.calculate(file)
            
            outputdir = os.path.join('/home/christosf/AttProtos/experiments/', args.dataset, args.model, 'reconstruction')
            
            mel = X[instance]
            label_list = dataset.label_list

            # "lrp.alpha_1_beta_0"
            # "lrp.alpha_2_beta_1"
            #"lrp.sequential_preset_a" 

            analysis = [
                "lrp.alpha_1_beta_0"               
            ]

            analyzers = {}
            for a in analysis:
                analyzers[a] = create_analyzer(a, model, neuron_selection_mode='index')
                
            #predictions = list(model.predict(X[instance])[0])
            #mean = np.mean(predictions, axis=0)
            #maxv = np.max(mean)
            #class_most_activated = predictions.index(maxv)
            
            explanations_voc = {}
            for a in analysis:
                explanations_voc[a] = []
                inputs, explanation = analyzers[a].analyze(mel, 0)
                explanations_voc[a].append(explanation)

            t_lrp_voc = np.array(explanations_voc["lrp.alpha_1_beta_0" ])
            t_lrp_voc = np.clip(
                t_lrp_voc,
                a_min=1e-15,
                a_max=40
            )


            salience_voc = t_lrp_voc
            
            mask_voc = scipy.special.expit(salience_voc[0])
            mask_voc = np.log10(0.01+mask_voc)
            mask_voc = (mask_voc - mask_voc.min()) / (mask_voc.max() - mask_voc.min())
            
            speech = []
            for i in range(len(X[instance])):
                scaled = X[instance][i]
                inversed = scaler.inverse_transform(scaled.T)
                melspec = librosa.db_to_power(inversed)
                masked_spec_voc = melspec*mask_voc[i].T
                inverse_voc = nnls(features.mel_basis, masked_spec_voc)
                inverse_voc = np.power(inverse_voc, 1./2.0, out=inverse_voc)
                polar_speech = inverse_voc * np.cos(phase[i].T) + inverse_voc * np.sin(phase[i].T) * 1j
                speech_out = librosa.istft(polar_speech, hop_length=features.audio_hop, win_length=features.audio_win, n_fft=features.n_fft, window='hann', center=True)
                speech.append(speech_out)
                

            time_context = len(speech[0])
            overlap = int(time_context/2) - int(features.audio_hop/2)
            #overlap = features.audio_hop*(int((features.sequence_hop_time/features.sequence_time)*features.sequence_frames))
            #overlap = int((time_context/(features.sequence_frames/features.sequence_hop))*2)
            signal = overlapadd(speech, time_context, overlap, y)
            path = outputdir + '/' + file_name.split('.wav')[0]

            try:
                os.makedirs(path)    
                print("Directory " , path ,  " Created ")
            except FileExistsError:
                print("Directory " , path ,  " already exists") 

            sf.write(path + '/' + file_name, signal, 44100)
                
            instance = instance +1

if __name__ == "__main__":
    main()
    