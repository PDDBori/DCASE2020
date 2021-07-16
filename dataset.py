import os
import numpy as np
import glob
import argparse
import sys
import random

import librosa
import librosa.core
import librosa.feature
import tensorflow as tf

import logging

import utils

from tqdm import tqdm


def file_list_generator(target_dir,
                        dir_name='train',
                        ext='wav'):
    print('target_dir : {}'.format(target_dir))
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir,
                                                                           dir_name=dir_name,
                                                                           ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        print("no_wav_file!!")
    print("train_file num : {num}".format(num=len(files)))
    return files


def file_load(file_name):
    try:
        return librosa.load(file_name, sr=None, mono=False)
    except:
        print("file_broken or not exists!! : {}".format(file_name))


def file_to_log_mel_spectrogram(param,
                                file_name):
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=param['feature']['n_fft'],
                                                     hop_length=param['feature']['hop_length'],
                                                     n_mels=param['feature']['n_mels'],
                                                     power=param['feature']['power'])
    log_mel = 20.0 / param['feature']['power'] * np.log10(mel_spectrogram + sys.float_info.epsilon + 0.0001)
    return log_mel


def find_id(file_name):
    for id_num in range(10):
        string = 'id_0' + str(id_num)
        if string in file_name:
            return id_num
        else:
            pass
    return -1


def list_to_log_mel_dataset(param,
                            file_list):
    idx_list = []
    for ii in tqdm(range(len(file_list)), desc="generate dataset"):
        log_mel = file_to_log_mel_spectrogram(param=param,
                                              file_name=file_list[ii])
        idx = find_id(file_list[ii])
        idx_list.append(idx)
        if ii == 0:
            log_mel_dataset = np.zeros((len(file_list), len(log_mel[:, 0]), len(log_mel[0, :]), 1), float)
        log_mel_dataset[ii, :, :, 0] = log_mel
    idx_list = np.array(idx_list)

    return log_mel_dataset, idx_list


def dataset_freq_normalizing(param,
                             dataset,
                             machine):
    if param['normalize'] == 0:
        return dataset
    else:
        if os.path.isfile('train_dataset_{}_mean.npy'.format(machine)):
            mean = np.load('train_dataset_{}_mean.npy'.format(machine))
        else:
            mean = np.mean(dataset, axis=(0, 2, 3))
            np.save('train_dataset_{}_mean.npy'.format(machine), mean)

        if os.path.isfile('train_dataset_{}_std.npy'.format(machine)):
            std = np.load('train_dataset_{}_std.npy'.format(machine))
        else:
            std = np.std(dataset, axis=(0, 2, 3))
            np.save('train_dataset_{}_std.npy'.format(machine), std)
        for ii in range(dataset.shape[1]):
            dataset[:, ii, :, :] = (dataset[:, ii, :, :] - mean[ii]) / std[ii]
        return dataset


def load_dataset(param,
                 machine,
                 dataset_dir):
    if os.path.isfile('train_dataset_{}.npy'.format(machine)) and os.path.isfile('train_dataset_{}_label.npy'.format(machine)):
        train_data = np.load('train_dataset_{}.npy'.format(machine))
        train_label = np.load('train_dataset_{}_label.npy'.format(machine))
    else:
        file_list = file_list_generator(dataset_dir)
        train_data, train_label = list_to_log_mel_dataset(param, file_list)
        np.save('train_dataset_{}.npy'.format(machine), train_data)
        np.save('train_dataset_{}_label.npy'.format(machine), train_label)
    return train_data, train_label


def label_modification(machine,
                       label):
    if machine == 'ToyCar' or machine == 'ToyConveyor':
        label = label - 1
    else:
        label = label
    return label


def dataset_windowing(param,
                      dataset):
    data_num = dataset.shape[0]
    time_length = dataset.shape[2]
    out_t = param['data_aug']['time_size']
    hop_size = param['data_aug']['hop']
    n_mel = param['feature']['n_mels']
    data_output = []
    for ii in range(data_num):
        idx = 0
        while idx + out_t < time_length:
            window = dataset[ii, :, idx:idx + out_t, :]
            data_output.append(window)
            idx += hop_size
    output = np.zeros((int(idx / hop_size) * data_num, n_mel, out_t, 1), float)
    for jj in range(len(data_output)):
        output[jj] = data_output[jj]

    return output


def dataset_freq_cut(dataset,
                     cut_freq):
    output = dataset[:, :, 0:cut_freq, :]
    return output


def dataset_preprocessing(param,
                          dataset,
                          label,
                          machine,
                          mode):
    dataset = dataset_freq_normalizing(param=param,
                                       dataset=dataset,
                                       machine=machine)

    # dataset = dataset_windowing(param=param,
    #                             dataset=dataset)

    # dataset = dataset_freq_cut(dataset,
    #                            cut_freq=312)

    if mode == 0:
        dataset = dataset.squeeze()
    dataset = np.moveaxis(dataset, 1, 2)

    return dataset


def label_mixing(param,
                 dataset,
                 label):
    num_data = dataset.shape[0]
    total_data = num_data * param['data_aug']['label_mix']
    output_dataset = []
    output_label = []

    for ii in tqdm(range(total_data)):
        d1 = random.randint(0, num_data - 1)
        d2 = random.randint(0, num_data - 1)
        data1 = dataset[d1]
        data2 = dataset[d2]
        label1 = label[d1]
        label2 = label[d2]
        w1 = random.random()
        w2 = 1 - w1
        data = w1 * data1 + w2 * data2
        data_label = (w1 * label1 + w2 * label2)

        output_dataset.append(data)
        output_label.append(data_label)

    return output_dataset, output_label


def two_data_linear_mixing(data1,
                           data2):
    time_frame = data1.shape[0]
    freq_frame = data1.shape[1]
    output_data = np.zeros((time_frame, freq_frame))

    for ii in range(time_frame):
        alpha = ii / (time_frame - 1)
        output_data[ii, :] = alpha * data1[ii, :] + (1 - alpha) * data2[ii, :]

    return output_data


def dataset_changing(param,
                     dataset,
                     label):
    num_data = dataset.shape[0]
    total_data = num_data * param['data_aug']['data_change']
    output_dataset = []
    output_label = []

    for ii in tqdm(range(total_data)):
        d1 = random.randint(0, num_data - 1)
        data1 = dataset[d1]
        label1 = label[d1]
        d2 = random.randint(0, num_data - 1)
        label2 = label[d2]
        while np.argmax(label1) != np.argmax(label2):
            d2 = random.randint(0, num_data - 1)
            label2 = label[d2]
        data2 = dataset[d2]

        data = two_data_linear_mixing(data1, data2)
        data_label = label1

        output_dataset.append(data)
        output_label.append(data_label)

    return output_dataset, output_label


def dataset_augmentation(param,
                         dataset,
                         label):
    # total dataset and label
    total_dataset = []
    total_label = []

    num_data = dataset.shape[0]

    # append original dataset
    for jj in range(num_data):
        total_dataset.append(dataset[jj])
        total_label.append(label[jj])

    # aug 1: label mixing
    print('Data Augmentation 1: label mixing')
    aug_data, aug_label = label_mixing(param, dataset, label)
    total_dataset = total_dataset + aug_data
    total_label = total_label + aug_label

    # aug 2: two data linear mixing
    print('Data Augmentation 2: linear mixing')
    aug_data, aug_label = dataset_changing(param, dataset, label)
    total_dataset = total_dataset + aug_data
    total_label = total_label + aug_label

    # change list to array
    total_dataset = np.array(total_dataset)
    total_label = np.array(total_label)

    return total_dataset, total_label


def generate_dataset(param,
                     machine,
                     dataset_dir,
                     mode=0):
    # load dataset
    train_data, train_label = load_dataset(param, machine, dataset_dir)

    # label modify
    train_label = label_modification(machine, train_label)

    # change label to one hot vector
    label = np.zeros((train_label.size, train_label.max() + 1))
    label[np.arange(train_label.size), train_label] = 1

    # dataset preprocessing
    train_data = dataset_preprocessing(param, train_data, label, machine, mode)

    # data augmentation
    train_data, label = dataset_augmentation(param, train_data, label)

    # make dataset tensor
    train_data = tf.data.Dataset.from_tensor_slices((train_data, label))\
        .shuffle(len(label))\
        .batch(param['fit']['batch_size'])

    num_batch = len(list(train_data))

    train = train_data.take(int(num_batch * 0.9))
    validation = train_data.skip(int(num_batch * 0.9))

    return train, validation


def get_mean_cov(param,
                 machine_type):
    train_dataset = np.load('train_dataset_{}.npy'.format(machine_type))
    train_dataset = dataset_freq_normalizing(param, train_dataset, machine_type)
    train_dataset = np.squeeze(train_dataset)

    # calculate mean
    train_dataset = np.mean(train_dataset, axis=2)
    mean = np.mean(train_dataset, axis=0)

    covariance = np.cov(np.transpose(train_dataset))

    return mean, covariance
