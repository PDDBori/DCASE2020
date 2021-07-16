import os
import sys
import numpy as np
import csv
from tqdm import tqdm
import tensorflow as tf
from sklearn import metrics
from scipy.spatial import distance

import utils
import model
import dataset

import matplotlib.pyplot as plt


def save_csv(save_file_path,
             save_data):
    """
    save score data
    :param save_file_path:
    :param save_data:
    :return:
    """
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def wn_anomaly_score_first(data,
                           model1,
                           model2):
    # get output data
    data = data.reshape((1, data.shape[0], data.shape[1]))
    tmp1 = model1.predict(data)
    reconstruction = model2.predict(tmp1)

    receptive_field = data.shape[1] - reconstruction.shape[1]

    # get error of data
    error = data[:, receptive_field:, :] - reconstruction
    # error = reconstruction - data[:, receptive_field:, :]

    error = np.squeeze(error)  # error shape: (T, F)
    error = np.square(error)

    # freq_bin = np.mean(error, axis=0)
    freq_bin = np.percentile(error, 95, axis=0)
    time_bin = np.percentile(freq_bin, 95)

    # loss1 = np.mean(np.square(error))

    total_loss = time_bin

    return total_loss


def wn_anomaly_score_second(data,
                            model1,
                            model2,
                            mean,
                            cov):
    data = data.reshape((1, data.shape[0], data.shape[1]))
    tmp1 = model1.predict(data)
    reconstruction = model2.predict(tmp1)

    receptive_field = data.shape[1] - reconstruction.shape[1]
    # get error of data
    error = data[:, receptive_field:, :] - reconstruction
    # error = reconstruction - data[:, receptive_field:, :]

    error = np.squeeze(error)  # error shape: (T, F)
    # error = np.square(error)
    error_freq = np.mean(error, axis=0)

    inv_cov = np.linalg.inv(cov)
    dist = distance.mahalanobis(error_freq, mean, inv_cov)

    total_error = dist

    return total_error


def wn_anomaly_score_third(data,
                           model1,
                           model2):
    """
    calculate reconstruction error of data
    :param data: input spectrogram (shape: Freq x Time)
    :param model1: residual blocks (input: Batch x Freq x Time, output: Batch x Channel X Time x Blocks)
    :param model2: reconstruction model (input: Batch x Freq x Time)
    :return: reconstruction error
    """
    data = data.reshape((1, data.shape[0], data.shape[1]))
    tmp1 = model1.predict(data)
    reconstruction = model2.predict(tmp1)
    receptive_field = data.shape[1] - reconstruction.shape[1]
    error = data[:, receptive_field:, :] - reconstruction
    loss1 = np.mean(np.square(error))
    return loss1


def wn_anomaly_score(machine_type,
                     data,
                     model1,
                     model2,
                     dataset_mean,
                     dataset_cov):
    if machine_type == 'valve' or machine_type == 'slider':
        # score = wn_anomaly_score_first(data, model1, model2)
        score = wn_anomaly_score_third(data, model1, model2)
    elif machine_type == 'ToyCar':
        score = wn_anomaly_score_second(data, model1, model2, dataset_mean, dataset_cov)
    elif machine_type == 'ToyConveyor':
        identity = np.identity(dataset_mean.shape[0])
        dataset_cov = np.multiply(dataset_cov, identity)
        score = wn_anomaly_score_second(data, model1, model2, dataset_mean, dataset_cov)
    else:
        dataset_cov = np.identity(dataset_mean.shape[0])
        score = wn_anomaly_score_second(data, model1, model2, dataset_mean, dataset_cov)

    return score


def main(param):
    mode = utils.mode_check(param)
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param['result_root'] + '/' + param['result_dir'], exist_ok=True)

    # load base directory
    data_dirs = utils.select_dirs(param=param, mode=mode)

    # select test dataset
    dirs = utils.select_machine(param, data_dirs)

    epoch_list = list(range(param['fit']['save_epoch'], param['fit']['epochs'] + 1, param['fit']['save_epoch']))
    # epoch_list = [300]

    epoch_id_AUC = []
    epoch_average_AUC = []

    # calculate AUCs for every epochs
    for epoch in epoch_list:
        # initialize lines in csv for AUC and pAUC
        csv_lines = []

        # repeat for each machine type
        for idx, target_dir in enumerate(dirs):
            print('\n===========================================================')
            print('[{idx}/{total}] {dirname}'.format(idx=idx + 1, total=len(dirs), dirname=target_dir))
            machine_type = os.path.split(target_dir)[1]

            id_AUC = []

            # load model
            tf.keras.backend.clear_session()
            model1 = model.model1_load(param, machine_type, epoch)
            model2 = model.model2_load(param, machine_type, epoch)

            # get mean and covariance of train machine dataset
            dataset_mean, dataset_cov = dataset.get_mean_cov(param, machine_type)

            # results by type for development data
            if mode:
                csv_lines.append([machine_type])
                csv_lines.append(["id", "AUC", "pAUC"])
                performance = []

            machine_id_list = utils.get_machine_id_list_for_test(target_dir)

            # repeat for each machine id
            for id_str in machine_id_list:
                # load test files
                test_files, y_true = utils.test_file_list_generator(target_dir, id_str, mode)

                # setup anomaly score file path
                anomaly_score_csv = '{root}/{result}/epoch_{ep}/anomaly_score_{machine}_{id}.csv'.format(root=param['result_root'],
                                                                                                         result=param['result_dir'],
                                                                                                         ep=epoch,
                                                                                                         machine=machine_type,
                                                                                                         id=id_str)

                # initialize anomaly score list
                anomaly_score_list = []

                print("\n============== BEGIN TEST FOR A MACHINE ID ==============")

                # initialize data labels
                y_pred = [0. for k in test_files]

                # get test dataset
                if os.path.isfile('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str)):
                    test_dataset = np.load('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str))
                else:
                    test_dataset, test_label = dataset.list_to_log_mel_dataset(param=param, file_list=test_files)
                    np.save('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str), test_dataset)

                # test dataset preprocessing
                test_dataset = dataset.dataset_preprocessing(param=param,
                                                             dataset=test_dataset,
                                                             label=0,
                                                             machine=machine_type,
                                                             mode=0)

                # calculate anomaly score for every data
                for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                    try:
                        # get data
                        data = test_dataset[file_idx]

                        # calculate loss
                        loss = wn_anomaly_score(machine_type, data, model1, model2, dataset_mean, dataset_cov)

                        y_pred[file_idx] = loss
                        anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    except:
                        print('file broken!!: {}'.format(file_path))

                # plot ROC curve
                plt.figure(figsize=(10, 8))
                fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
                plt.plot(fpr, tpr)
                plt.title('{type} {id}'.format(type=machine_type, id=id_str))
                os.makedirs(param['result_root'] + '/' + param['result_dir'] + '/epoch_' + str(epoch), exist_ok=True)
                plt.savefig('{root}/{model}/epoch_{ep}/{mt}_{id}'.format(root=param['result_root'], model=param['result_dir'], ep=epoch, mt=machine_type, id=id_str))

                # save anomaly score
                save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)

                # append AUC and pAUC to lists for development data
                if mode:
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param['max_fpr'])
                    csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                    performance.append([auc, p_auc])
                    id_AUC.append(auc)
                    print('AUC: {}'.format(auc))
                    print('pAUC: {}'.format(p_auc))
                print("\n============ END OF TEST FOR A MACHINE ID ============")

            # calculate averages for AUCs and pAUCs for development data
            if mode:
                averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
                csv_lines.append(["Average"] + list(averaged_performance))
                csv_lines.append([])

                epoch_id_AUC.append(id_AUC)
                epoch_average_AUC.append(np.mean(np.array(id_AUC)))

        # output results for development data
        if mode:
            result_path = '{root}/{result}/epoch_{ep}/{file_name}'.format(root=param['result_root'],
                                                                          result=param['result_dir'],
                                                                          ep=epoch,
                                                                          file_name=param['result_file'])
            print("AUC and pAUC results -> {}".format(result_path))
            save_csv(save_file_path=result_path,
                     save_data=csv_lines)

    for idx, target_dir in enumerate(dirs):
        machine_type = os.path.split(target_dir)[1]
        machine_id_list = utils.get_machine_id_list_for_test(target_dir)

        idx_AUC = np.array(epoch_id_AUC[idx::len(dirs)])
        av_AUC = np.array(epoch_average_AUC[idx::len(dirs)])

        plt.figure(figsize=(15, 8))
        for ind, jj in enumerate(machine_id_list):
            plt.plot(epoch_list, idx_AUC[:, ind])
        plt.legend(machine_id_list, loc='upper right', fontsize=15)
        plt.title('{} AUC'.format(machine_type), fontsize=20)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('AUC', fontsize=15)
        plt.tick_params(axis='both', labelsize=12)
        plt.xticks(epoch_list)
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.ylim(0.45, 1.05)
        plt.grid(True)
        plt.savefig('{root}/{model}/{machine}_id'.format(root=param['result_root'], model=param['result_dir'], machine=machine_type))

        plt.figure(figsize=(15, 8))
        plt.plot(epoch_list, av_AUC)
        plt.legend([machine_type], loc='upper right', fontsize=15)
        plt.title('{} AUC'.format(machine_type), fontsize=20)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('AUC', fontsize=15)
        plt.tick_params(axis='both', labelsize=12)
        plt.xticks(epoch_list)
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.ylim(0.45, 1.05)
        plt.grid(True)
        plt.savefig('{root}/{model}/{machine}'.format(root=param['result_root'], model=param['result_dir'],
                                                      machine=machine_type))


if __name__ == '__main__':
    param_ = utils.yaml_load()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(param_['gpu_num'])
    print(param_['result_dir'])
    main(param_)
