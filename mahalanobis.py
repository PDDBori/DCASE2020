import os
import sys
import numpy as np
import tensorflow as tf

import utils
import dataset
import model
import test
from scipy.spatial import distance
from tqdm import tqdm


def get_loss_vector(machine_type, train_dataset, reconstruction, dataset_mean, dataset_cov):
    data_num = train_dataset.shape[0]
    data_1 = train_dataset.shape[1]
    data_2 = train_dataset.shape[2]

    recon_1 = reconstruction.shape[1]
    recon_2 = reconstruction.shape[2]

    total_loss = []
    for ii in tqdm(range(data_num)):
        input_data = train_dataset[ii].reshape((1, data_1, data_2))
        recon = reconstruction[ii].reshape((1, recon_1, recon_2))
        loss = test.reconstruction_loss(machine_type, input_data, recon, dataset_mean, dataset_cov)
        total_loss.append(loss)
    total_loss = np.array(total_loss)

    return total_loss


def main(param):
    md_mode = utils.mode_check(param)
    if md_mode is None:
        sys.exit(-1)

    # load base_directory list
    data_dirs = utils.select_dirs(param, md_mode)

    # select training dataset
    dirs = utils.select_machine(param, data_dirs)

    for idx, target_dir in enumerate(dirs):
        print('\n===========================================================')
        print('[{idx}/{total}] {dirname}'.format(idx=idx + 1, total=len(dirs), dirname=target_dir))

        machine_type = os.path.split(target_dir)[1]

        # get mean and covariance of train machine dataset
        dataset_mean, dataset_cov = dataset.get_mean_cov(param, machine_type)

        print("============== DATASET_GENERATOR ==============")
        train_dataset = np.load('train_dataset_{}.npy'.format(machine_type))
        train_label = np.load('train_dataset_{}_label.npy'.format(machine_type))

        train_label = dataset.label_modification(machine_type, train_label)

        # make label to one hot vector
        label = np.zeros((train_label.size, train_label.max() + 1))
        label[np.arange(train_label.size), train_label] = 1

        print("============== DATASET_PREPROCESSING ==============")
        train_dataset = dataset.dataset_preprocessing(param=param,
                                                      dataset=train_dataset,
                                                      label=label,
                                                      machine=machine_type,
                                                      mode=0)

        # epoch_list = list(range(param['fit']['save_epoch'], param['fit']['epochs'] + 1, param['fit']['save_epoch']))
        epoch_list = [300]

        for epoch in epoch_list:
            # get model
            tf.keras.backend.clear_session()
            model1 = model.model1_load(param, machine_type, epoch)
            model2 = model.model2_load(param, machine_type, epoch)
            model3 = model.model3_load(param, machine_type, epoch)

            print("============== GET_MAHALANOBIS_DISTANCE ==============")
            loss1 = []
            loss2 = []
            for ii in tqdm(range(len(train_label))):
                input_data = np.expand_dims(train_dataset[ii], 0)
                tmp = model1.predict(input_data)
                tmp1 = model2.predict(tmp)
                tmp2 = model3.predict(tmp)


                data_loss1 = test.reconstruction_loss(machine_type, input_data, tmp1, dataset_mean, dataset_cov)
                data_loss2 = tf.keras.losses.categorical_crossentropy(label[ii], tmp2[0]).numpy()

                loss1.append(data_loss1)
                loss2.append(data_loss2)

            loss1 = np.array(loss1)
            loss2 = np.array(loss2)

            loss1_mean = np.mean(loss1)
            loss2_mean = np.mean(np.exp(-loss2))

            mean = np.array((loss1_mean, loss2_mean))
            cov = np.cov(loss1, np.exp(-loss2))

        # print("============== GET_MAHALANOBIS_DISTANCE ==============")
        # tmp_data = model1.predict(train_dataset)
        # reconstruction = model2.predict(tmp_data)
        # classification = model3.predict(tmp_data)
        #
        # # calculate loss 1 (mse loss)
        # loss1 = get_loss_vector(machine_type, train_dataset, reconstruction, dataset_mean, dataset_cov)
        #
        # loss1_mean = np.mean(loss1)
        #
        # loss2 = tf.keras.losses.categorical_crossentropy(label, classification).numpy()
        # loss2_mean = np.mean(np.exp(-loss2))
        #
        # mean = np.array((loss1_mean, loss2_mean))
        # cov = np.cov(loss1, np.exp(-loss2))

        # save mean and covariance matrix
            np.save('{root}/{model}/epoch_{ep}/mean_{machine}'.format(root=param['model_root'], model=param['model_dir'], ep=epoch, machine=machine_type), mean)
            np.save('{root}/{model}/epoch_{ep}/cov_{machine}'.format(root=param['model_root'], model=param['model_dir'], ep=epoch, machine=machine_type), cov)

            # print('{machine} end'.format(machine=machine_type))


if __name__ == '__main__':
    param_ = utils.yaml_load()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param_['gpu_num'])
    print(param_['model_dir'])
    main(param_)
