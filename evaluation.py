import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import sys

import tensorflow as tf

import dataset
import utils
import model


def get_dev_eval_str(num):
    if num == 0:
        output = 'dev_data'
    else:
        output = 'eval_data'
    return output


def get_machine_type_str(num):
    if num == 0:
        output = 'fan'
    elif num == 1:
        output = 'pump'
    elif num == 2:
        output = 'slider'
    elif num == 3:
        output = 'ToyCar'
    elif num == 4:
        output = 'ToyConveyor'
    else:
        output = 'valve'
    return output


def get_train_test_str(num):
    if num == 0:
        output = 'train'
    else:
        output = 'test'
    return output


def get_normal_anomaly_str(num):
    if num == 0:
        output = 'normal'
    else:
        output = 'anomaly'
    return output


def get_machine_id_str(num):
    output = 'id_0{}'.format(num)
    return output


def get_data_num_str(num):
    output = f'{num:08}'
    return output


def get_in_out_str(num):
    if num == 0:
        output = 'input'
    elif num == 1:
        output = 'output'
    elif num == 2:
        output = 'error'
    elif num == 3:
        output = 'error_input'
    elif num == 4:
        output = 'error_output'
    elif num == 5:
        output = 'error_diff'
    elif num == 10:
        output = 'original_input'
    else:
        output = 'error_diff'
    return output


def file_to_log_mel_spectrogram(param,
                                file_name):
    # generate mel_spectrogram using librosa
    y, sr = dataset.file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=param['feature']['n_fft'],
                                                     hop_length=param['feature']['hop_length'],
                                                     n_mels=param['feature']['n_mels'],
                                                     power=param['feature']['power'])
    # convert mel_spectrogram to log mel spectrogram
    log_mel_spectrogram = 20.0 / param['feature']['power'] * np.log10(mel_spectrogram + sys.float_info.epsilon + 0.0001)

    return log_mel_spectrogram


def one_data_normalizing(data, machine, idx):
    if idx == 0:
        return data
    else:
        mean = np.load('train_dataset_{}_mean.npy'.format(machine))
        std = np.load('train_dataset_{}_std.npy'.format(machine))

        for ii in range(data.shape[0]):
            data[ii, :] = (data[ii, :] - mean[ii]) / std[ii]
        return data


def one_data_preprocessing(data):
    data = data.reshape((1, data.shape[0], data.shape[1]))

    # output = np.zeros((1, data.shape[1], 312))
    # output[:, :, :] = data[:, :, 0:312]
    output = np.moveaxis(data, 1, 2)

    return output


def plot_figure0_original_input(fig, ma, de, tt, na, mi, dn, md):
    data = fig.squeeze()
    data = np.moveaxis(data, 0, 1)

    plt.figure(figsize=(25, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    librosa.display.specshow(data,
                             cmap=plt.get_cmap('magma'),
                             vmin=-5,
                             vmax=5,
                             y_axis='mel')
    figure_name = '{de} {tt} {na} {mi} #{dn} 0.original_input'.format(de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=30)
    plt.xlabel('Time Frame', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


def plot_figure1_input(fig, ma, de, tt, na, mi, dn, md, rf):
    data = fig.squeeze()
    data = np.moveaxis(data, 0, 1)
    data1 = data[:, rf:]

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(data1,
                             cmap=plt.get_cmap('magma'),
                             vmin=-5,
                             vmax=5,
                             y_axis='mel')
    figure_name = '{de} {tt} {na} {mi} #{dn} 1.input'.format(de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=30)
    plt.xlabel('Time Frame', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


def plot_figure2_output(fig, ma, de, tt, na, mi, dn, md):
    data = fig.squeeze()
    data = np.moveaxis(data, 0, 1)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(data,
                             cmap=plt.get_cmap('magma'),
                             vmin=-5,
                             vmax=5,
                             y_axis='mel')
    figure_name = '{de} {tt} {na} {mi} #{dn} 2.reconstruction'.format(de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=30)
    plt.xlabel('Time Frame', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


def plot_figure3_error(fig1, fig2, ma, de, tt, na, mi, dn, md, rf):
    data1 = fig1.squeeze()
    data2 = fig2.squeeze()

    input_fig = data1[rf:, :]
    error = input_fig - data2
    error = np.moveaxis(error, 0, 1)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(error,
                             cmap=plt.get_cmap('RdGy'),
                             vmin=-5,
                             vmax=5,
                             y_axis='mel')
    figure_name = '{de} {tt} {na} {mi} #{dn} 3.error'.format(de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=30)
    plt.xlabel('Time Frame', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


def plot_figure4_classification(data, ma, de, tt, na, mi, dn, md):
    y_prob = np.squeeze(data)
    if ma == 'ToyCar':
        x_label = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']
    elif ma == 'ToyConveyor':
        x_label = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
    else:
        x_label = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']

    plt.figure(figsize=(10, 10))
    x = np.arange(len(x_label))
    plt.bar(x, y_prob)
    plt.ylim(0, 1)
    plt.xticks(x, x_label, fontsize=20)
    figure_name = '{de} {tt} {na} {mi} #{dn} 4.classification'.format(de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=20)
    # plt.xlabel('ID', fontsize=30)
    plt.ylabel('Probability', fontsize=20)
    # plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == '__main__':
    #############################################################
    # get parameters
    model_dir = '2021-06-30 MTL mix0'
    # 0: not normalization / 1: normalization
    norm_idx = 1

    # 0: development / 1: evaluation
    dev_eval_idx = 0
    # 0: fan / 1: pump / 2: slider / 3: ToyCar / 4: ToyConveyor / 5: valve
    machine_type_idx = 3
    # 0: train / 1: test
    train_test_idx = 1
    # 0: normal / 1: anomaly
    normal_anomaly_idx = 1
    # 0: id_00 / 1: id_01 / 2: id_02 / 3: id_03 / 4: id_04 ...
    machine_id_idx = 2
    # data number
    data_num = [0, 1, 2, 3, 4]

    #############################################################

    param = utils.yaml_load()

    # get str of parameters
    des = get_dev_eval_str(dev_eval_idx)
    mts = get_machine_type_str(machine_type_idx)
    tts = get_train_test_str(train_test_idx)
    nas = get_normal_anomaly_str(normal_anomaly_idx)
    mis = get_machine_id_str(machine_id_idx)

    # set model path
    model1_path = './model/{model}/model1_{machine}'.format(model=model_dir, machine=mts)
    model2_path = './model/{model}/model2_{machine}'.format(model=model_dir, machine=mts)
    model3_path = './model/{model}/model3_{machine}'.format(model=model_dir, machine=mts)

    # get model
    model1 = tf.keras.models.load_model(model1_path)
    model2 = tf.keras.models.load_model(model2_path)
    model3 = tf.keras.models.load_model(model3_path)

    for idx in range(len(data_num)):

        dns = get_data_num_str(data_num[idx])

        # get sound data
        file_dir = './{d_e}/{m_t}/{t_t}/{n_a}_{m_i}_{d_n}.wav'.format(d_e=des, m_t=mts, t_t=tts, n_a=nas, m_i=mis, d_n=dns)

        # get log mel spectrogram
        log_mel = file_to_log_mel_spectrogram(param, file_dir)

        # data normalization
        norm_data = one_data_normalizing(log_mel, mts, norm_idx)

        # data preprocessing
        input_data = one_data_preprocessing(norm_data)

        global_feature = model1.predict(input_data)
        reconstruction = model2.predict(global_feature)
        classification = model3.predict(global_feature)

        rf = input_data.shape[1] - reconstruction.shape[1]

        plot_figure0_original_input(input_data, mts, des, tts, nas, mis, data_num[idx], model_dir)
        plot_figure1_input(input_data, mts, des, tts, nas, mis, data_num[idx], model_dir, rf)
        plot_figure2_output(reconstruction, mts, des, tts, nas, mis, data_num[idx], model_dir)
        plot_figure3_error(input_data, reconstruction, mts, des, tts, nas, mis, data_num[idx], model_dir, rf)
        plot_figure4_classification(classification, mts, des, tts, nas, mis, data_num[idx], model_dir)

    print('end')
