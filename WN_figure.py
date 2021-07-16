import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import sys

import tensorflow as tf

import dataset


def get_dev_eval_str(idx):
    if idx == 0:
        output = 'dev_data'
    else:
        output = 'eval_data'
    return output


def get_machine_type_str(idx):
    if idx == 0:
        output = 'fan'
    elif idx == 1:
        output = 'pump'
    elif idx == 2:
        output = 'slider'
    elif idx == 3:
        output = 'ToyCar'
    elif idx == 4:
        output = 'ToyConveyor'
    else:
        output = 'valve'
    return output


def get_train_test_str(idx):
    if idx == 0:
        output = 'train'
    else:
        output = 'test'
    return output


def get_normal_anomaly_str(idx):
    if idx == 0:
        output = 'normal'
    else:
        output = 'anomaly'
    return output


def get_machine_id_str(idx):
    output = 'id_0{}'.format(idx)
    return output


def get_data_num_str(idx):
    output = f'{idx:08}'
    return output


def get_in_out_str(idx):
    if idx == 0:
        output = 'input'
    elif idx == 1:
        output = 'output'
    elif idx == 2:
        output = 'error'
    elif idx == 3:
        output = 'error_input'
    elif idx == 4:
        output = 'error_output'
    elif idx == 5:
        output = 'error_diff'
    elif idx == 10:
        output = 'original_input'
    else:
        output = 'error_diff'
    return output


def file_to_log_mel_spectrogram(file_name,
                                n_mels=128,
                                n_fft=2048,
                                hop_length=512,
                                power=2.0):
    # generate mel_spectrogram using librosa
    y, sr = dataset.file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # convert mel_spectrogram to log mel spectrogram
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon + 0.0001)

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
    figure_name = '{ma} {de} {tt} {na} {mi} #{dn} 0.original_input'.format(ma=ma, de=de, tt=tt, na=na, mi=mi, dn=dn)
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
    figure_name = '{ma} {de} {tt} {na} {mi} #{dn} 1.input'.format(ma=ma, de=de, tt=tt, na=na, mi=mi, dn=dn)
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
    figure_name = '{ma} {de} {tt} {na} {mi} #{dn} 2.reconstruction'.format(ma=ma, de=de, tt=tt, na=na, mi=mi, dn=dn)
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
    figure_name = '{ma} {de} {tt} {na} {mi} #{dn} 3.error'.format(ma=ma, de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=30)
    plt.xlabel('Time Frame', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


def plot_figure4_classification(data, ma, de, tt, na, mi, dn, md):
    y_prob = np.squeeze(data)
    if ma == 'ToyCar' or ma == 'ToyConveyor':
        x_label = ['id_01', 'id_02', 'id_03', 'id_04']
    else:
        x_label = ['id_00', 'id_02', 'id_04', 'id_06']

    plt.figure(figsize=(10, 10))
    x = np.arange(4)
    plt.bar(x, y_prob)
    plt.ylim(0, 1)
    plt.xticks(x, x_label, fontsize=20)
    figure_name = '{ma} {de} {tt} {na} {mi} #{dn} 4.classification'.format(ma=ma, de=de, tt=tt, na=na, mi=mi, dn=dn)
    plt.title(figure_name, fontsize=20)
    # plt.xlabel('ID', fontsize=30)
    plt.ylabel('Probability', fontsize=20)
    # plt.colorbar(label='normalized')
    os.makedirs('./model/' + md + '/figures/' + ma, exist_ok=True)
    plt.savefig('./model/' + md + '/figures/' + ma + '/' + figure_name + '.png')


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    # get parameters
    model_dir = '2021-06-26 WN'
    # 0: not normalization / 1: normalization
    norm_idx = 1

    # 0: development / 1: evaluation
    dev_eval_idx = 0
    # 0: fan / 1: pump / 2: slider / 3: ToyCar / 4: ToyConveyor / 5: valve
    machine_type_idx = 5
    # 0: train / 1: test
    train_test_idx = 0
    # 0: normal / 1: anomaly
    normal_anomaly_idx = 0
    # 0: id_00 / 1: id_01 / 2: id_02 / 3: id_03 / 4: id_04 ...
    machine_id_idx = 0
    # data number
    data_num_idx = [0, 1, 2, 3, 4]
    # n fft
    nfft = 2048

    # get str of parameters
    dev_eval_str = get_dev_eval_str(dev_eval_idx)
    machine_type_str = get_machine_type_str(machine_type_idx)
    train_test_str = get_train_test_str(train_test_idx)
    normal_anomaly_str = get_normal_anomaly_str(normal_anomaly_idx)
    machine_id_str = get_machine_id_str(machine_id_idx)


    # set model path
    model1_path = './model/{model}/model1_{machine}'.format(model=model_dir, machine=machine_type_str)
    model2_path = './model/{model}/model2_{machine}'.format(model=model_dir, machine=machine_type_str)

    # get model
    model1 = tf.keras.models.load_model(model1_path)
    model2 = tf.keras.models.load_model(model2_path)

    for ii in range(len(data_num_idx)):
        data_num_str = get_data_num_str(data_num_idx[ii])

        # get sound data
        sound_file_dir = './{d_e}/{m_t}/{t_t}/{n_a}_{m_i}_{d_n}.wav'.format(d_e=dev_eval_str,
                                                                            m_t=machine_type_str,
                                                                            t_t=train_test_str,
                                                                            n_a=normal_anomaly_str,
                                                                            m_i=machine_id_str,
                                                                            d_n=data_num_str)
        # get log mel spectrogram
        sound_log_mel = file_to_log_mel_spectrogram(sound_file_dir, n_fft=nfft)

        norm_data = one_data_normalizing(sound_log_mel, machine_type_str, norm_idx)

        input_data = one_data_preprocessing(norm_data)



        global_feature = model1.predict(input_data)
        reconstruction = model2.predict(global_feature)

        receptive_field = input_data.shape[1] - reconstruction.shape[1]

        plot_figure0_original_input(input_data, machine_type_str, dev_eval_str, train_test_str, normal_anomaly_str, machine_id_str, data_num_idx[ii], model_dir)
        plot_figure1_input(input_data, machine_type_str, dev_eval_str, train_test_str, normal_anomaly_str, machine_id_str, data_num_idx[ii], model_dir, receptive_field)
        plot_figure2_output(reconstruction, machine_type_str, dev_eval_str, train_test_str, normal_anomaly_str, machine_id_str, data_num_idx[ii], model_dir)
        plot_figure3_error(input_data, reconstruction, machine_type_str, dev_eval_str, train_test_str, normal_anomaly_str, machine_id_str, data_num_idx[ii], model_dir, receptive_field)

    print('end')
