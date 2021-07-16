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
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def reconstruction_loss1(data, reconstruction):
    # get error map
    receptive_field = data.shape[1] - reconstruction.shape[1]
    error = data[:, receptive_field:, :] - reconstruction
    error = np.squeeze(error)  # error shape: (T, F)
    error = np.square(error)

    freq_bin = np.percentile(error, 95, axis=0)
    time_bin = np.percentile(freq_bin, 95)

    # loss = np.mean(np.square(error), axis=(1, 2))[0]

    loss = time_bin

    return loss


def reconstruction_loss2(data, reconstruction, dataset_mean, dataset_cov):
    # get error map
    receptive_field = data.shape[1] - reconstruction.shape[1]
    error = data[:, receptive_field:, :] - reconstruction
    error = np.squeeze(error)  # error shape: (T, F)
    # error = np.square(error)

    # get freq bin vector
    error_freq = np.mean(error, axis=0)

    # get inverse covariance
    dataset_inv_cov = np.linalg.inv(dataset_cov)
    dist = distance.mahalanobis(error_freq, dataset_mean, dataset_inv_cov)

    loss = dist

    return loss


def reconstruction_loss3(data, reconstruction):
    receptive_field = data.shape[1] - reconstruction.shape[1]
    error = data[:, receptive_field:, :] - reconstruction
    loss = np.mean(np.square(error), axis=(1, 2))[0]

    return loss


def classification_loss(label, predict):
    loss = tf.keras.losses.categorical_crossentropy(label, predict).numpy()
    return loss


def reconstruction_loss(machine_type, data, reconstruction, dataset_mean, dataset_cov):
    # calculate loss 1 (mse loss)
    if machine_type == 'slider':
        # loss1 = reconstruction_loss1(data, reconstruction)
        loss1 = reconstruction_loss3(data, reconstruction)

    elif machine_type == 'valve':
        # loss1 = reconstruction_loss1(data, reconstruction)
        loss1 = reconstruction_loss3(data, reconstruction)

    elif machine_type == 'ToyCar':
        loss1 = reconstruction_loss2(data, reconstruction, dataset_mean, dataset_cov)
        # loss1 = reconstruction_loss3(data, reconstruction)

    elif machine_type == 'ToyConveyor':
        identity = np.identity(dataset_mean.shape[0])
        dataset_cov = np.multiply(dataset_cov, identity)
        loss1 = reconstruction_loss2(data, reconstruction, dataset_mean, dataset_cov)
        # loss1 = reconstruction_loss3(data, reconstruction)

    else:
        dataset_cov = np.identity(dataset_mean.shape[0])
        loss1 = reconstruction_loss2(data, reconstruction, dataset_mean, dataset_cov)
        # loss1 = reconstruction_loss3(data, reconstruction)

    return loss1


def mtl_anomaly_score(machine_type, data, label, model1, model2, model3, loss_mean, loss_inv_cov, dataset_mean, dataset_cov):

    data = data.reshape((1, data.shape[0], data.shape[1]))
    tmp = model1(data)
    reconstruction = model2(tmp)
    classification = model3(tmp)

    # calculate loss 1 (mse loss)
    loss1 = reconstruction_loss(machine_type, data, reconstruction, dataset_mean, dataset_cov)

    # calculate loss 2 (cross entropy loss)
    loss2 = classification_loss(label, classification[0])

    # create md array
    input_vector = np.array((loss1, np.exp(-loss2)))

    # calculate mahalanobis distance
    total = distance.mahalanobis(input_vector, loss_mean, loss_inv_cov)

    return loss1, loss2, total


def main(param):
    test_mode = utils.mode_check(param)
    if test_mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param['result_root'] + '/' + param['result_dir'], exist_ok=True)

    # load base directory
    data_dirs = utils.select_dirs(param=param, mode=test_mode)

    # select test dataset
    dirs = utils.select_machine(param, data_dirs)

    # epoch_list = list(range(param['fit']['save_epoch'], param['fit']['epochs'] + 1, param['fit']['save_epoch']))
    epoch_list = [300]

    epoch_id_AUC = []
    epoch_average_AUC = []

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
            model3 = model.model3_load(param, machine_type, epoch)

            # get mean and covariance of train machine dataset
            dataset_mean, dataset_cov = dataset.get_mean_cov(param, machine_type)

            # results by type for development data
            if test_mode:
                csv_lines.append([machine_type])
                csv_lines.append(["id", "AUC", "pAUC"])
                performance = []

            machine_id_list = utils.get_machine_id_list_for_test(target_dir)

            # repeat for each machine id
            for id_str in machine_id_list:

                # load test files
                test_files, y_true = utils.test_file_list_generator(target_dir, id_str, test_mode)

                # setup anomaly score file path
                anomaly_score_csv = '{root}/{result}/epoch_{ep}/anomaly_score_{machine}_{id}.csv'.format(root=param['result_root'], result=param['result_dir'], ep=epoch, machine=machine_type, id=id_str)

                # initialize anomaly score list
                anomaly_score_list = []

                print("\n============== BEGIN TEST FOR A MACHINE ID ==============")

                # initialize data labels
                y_pred = [0. for k in test_files]

                # get test dataset
                if os.path.isfile('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str)):
                    test_dataset = np.load('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str))
                else:
                    test_dataset = dataset.list_to_log_mel_dataset(param=param, file_list=test_files)
                    np.save('test_dataset_{machine}_{id}.npy'.format(machine=machine_type, id=id_str), test_dataset)

                # test dataset preprocessing
                test_dataset = dataset.dataset_preprocessing(param=param, dataset=test_dataset, label=0, machine=machine_type, mode=0)

                # load mean and covariance matrix
                loss_mean = np.load('{root}/{model}/epoch_{ep}/mean_{machine}.npy'.format(root=param['model_root'], model=param['model_dir'], ep=epoch, machine=machine_type))
                loss_cov = np.load('{root}/{model}/epoch_{ep}/cov_{machine}.npy'.format(root=param['model_root'], model=param['model_dir'], ep=epoch, machine=machine_type))
                loss_inv_cov = np.linalg.inv(loss_cov)

                # calculate anomaly score for every data
                for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                    try:
                        # get data
                        data = test_dataset[file_idx]

                        # get label
                        a = int(list(id_str)[4])
                        if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
                            a = a - 1
                        if machine_type == 'ToyConveyor':
                            label = np.zeros(6)
                        else:
                            label = np.zeros(7)
                        label[a] = 1

                        # calculate total loss
                        loss1, loss2, total_loss = mtl_anomaly_score(machine_type, data, label, model1, model2, model3, loss_mean, loss_inv_cov, dataset_mean, dataset_cov)

                        y_pred[file_idx] = total_loss
                        anomaly_score_list.append(
                            [os.path.basename(file_path), total_loss, loss1, loss2, np.exp(-loss2)])
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
                if test_mode:
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param['max_fpr'])
                    csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                    performance.append([auc, p_auc])
                    id_AUC.append(auc)
                    print('AUC: {}'.format(auc))
                    print('pAUC: {}'.format(p_auc))
                print("\n============ END OF TEST FOR A MACHINE ID ============")

            # calculate averages for AUCs and pAUCs for development data
            if test_mode:
                averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
                csv_lines.append(["Average"] + list(averaged_performance))
                csv_lines.append([])

                epoch_id_AUC.append(id_AUC)
                epoch_average_AUC.append(np.mean(np.array(id_AUC)))

        # output results for development data
        if test_mode:
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param_['gpu_num'])
    print(param_['result_dir'])
    main(param_)
