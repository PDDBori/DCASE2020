import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import sys
import numpy as np
import time
import tensorflow as tf

import utils
import dataset
import model
import test
import matplotlib.pyplot as plt
from tqdm import tqdm


class Visualizer(object):
    def __init__(self):
        self.legend = ['train', 'validation']
        self.machine_type = []
        self.loss1_train = []
        self.loss1_val = []
        self.loss2_train = []
        self.loss2_val = []

    def add_machine_type(self, machine_type):
        self.machine_type.append(machine_type)

    def add_train_loss1(self, train_loss):
        self.loss1_train.append(train_loss)

    def add_val_loss1(self, val_loss):
        self.loss1_val.append(val_loss)

    def add_train_loss2(self, train_loss):
        self.loss2_train.append(train_loss)

    def add_val_loss2(self, val_loss):
        self.loss2_val.append(val_loss)

    def plot_loss(self, loss1_train, loss1_val, loss2_train, loss2_val):
        fig = plt.figure(figsize=(30, 20))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        figure1 = fig.add_subplot(2, 1, 1)
        figure2 = fig.add_subplot(2, 1, 2)

        self.plot_loss1(figure1, loss1_train, loss1_val)
        self.plot_loss2(figure2, loss2_train, loss2_val)
        figure1.legend(self.legend, loc='upper right', fontsize=20)
        figure2.legend(self.legend, loc='upper right', fontsize=20)

    def plot_loss1(self, figure1, train, val):
        figure1.plot(np.log(train))
        figure1.plot(np.log(val))
        figure1.set_title("{} loss 1".format(self.machine_type[0]), fontsize=30)
        figure1.set_xlabel("Epoch", fontsize=30)
        figure1.set_ylabel("Reconstruction Loss (ln)", fontsize=30)
        figure1.set_ylim([-3, 1])
        figure1.tick_params(axis='both', labelsize=20)

        figure1.grid()

    def plot_loss2(self, figure2, train, val):
        figure2.plot(train)
        figure2.plot(val)
        figure2.set_title("{} loss 2".format(self.machine_type[0]), fontsize=30)
        figure2.set_xlabel("Epoch", fontsize=30)
        figure2.set_ylabel("Cross-Entropy Loss", fontsize=30)
        figure2.set_ylim([0, 1])
        figure2.tick_params(axis='both', labelsize=20)

        figure2.grid()

    def save_figure(self, name):
        self.plot_loss(self.loss1_train, self.loss1_val, self.loss2_train, self.loss2_val)
        plt.savefig(name)


def main(param):
    train_mode = utils.mode_check(param)
    if train_mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(param['model_root'] + '/' + param['model_dir'], exist_ok=True)

    # load base_directory list
    data_dirs = utils.select_dirs(param, train_mode)

    # select training dataset
    dirs = utils.select_machine(param, data_dirs)

    # repeat for each machine directory
    for idx, target_dir in enumerate(dirs):
        print('\n===========================================================')
        print('[{idx}/{total}] {dirname}'.format(idx=idx + 1, total=len(dirs), dirname=target_dir))

        machine_type = os.path.split(target_dir)[1]

        # initialize the visualizer
        visualizer = Visualizer()

        visualizer.add_machine_type(machine_type)

        print("============== DATASET_GENERATOR ==============")
        train_dataset, validation_dataset = dataset.generate_dataset(param=param, machine=machine_type, dataset_dir=target_dir)

        print("============== MODEL_TRAINING ==============")
        # set model parameters
        input_size = list(train_dataset)[0][0].shape[1]
        num_channel = param['model']['num_channel']
        num_filters = param['model']['channel_multiply'] * num_channel
        kernel_size = param['model']['kernel_size']
        num_blocks = param['model']['num_blocks']
        num_classes = list(train_dataset)[0][1].shape[1]

        train_model1 = model.Wavenet_blocks(input_size, num_channel, num_filters, kernel_size, num_blocks)
        train_model2 = model.Wavenet_reconstruction(input_size, num_channel, kernel_size, num_blocks)
        train_model3 = model.Resnet18(input_size, num_channel, num_blocks, n_output=num_classes)

        # get receptive field of WaveNet
        receptive_field = train_model2.receptive_field

        # for each epoch
        for epoch in range(param['fit']['epochs']):
            # start time
            start_time = time.time()

            # initialize optimizers
            optimizer_1 = tf.keras.optimizers.Adam(param['learning_rate']['model1'], beta_1=0.85)
            optimizer_2 = tf.keras.optimizers.Adam(param['learning_rate']['model2'], beta_1=0.85)
            optimizer_3 = tf.keras.optimizers.Adam(param['learning_rate']['model3'], beta_1=0.85)

            # first batch index
            batch_idx = 0

            # initialize validation loss
            tmp_val_loss1 = []
            tmp_val_loss2 = []

            # for each train batch
            for data, label in train_dataset:
                with tf.GradientTape() as m1_tape, tf.GradientTape() as m2_tape, tf.GradientTape() as m3_tape:
                    skip = train_model1(data)
                    reconstruction = train_model2(skip)
                    classifier = train_model3(skip)

                    # calculate loss 1
                    tmp = tf.cast(data[:, receptive_field:, :], dtype=tf.float32)
                    mse = tf.square(tf.subtract(reconstruction, tmp))
                    tmp3 = tf.reduce_mean(mse, axis=[1, 2])
                    train_loss1 = tf.reduce_mean(tmp3)

                    # calculate loss 2
                    entropy_loss = tf.keras.losses.categorical_crossentropy(label, classifier)
                    train_loss2 = tf.reduce_mean(entropy_loss)

                    # calculate total loss
                    if train_loss2 > 1e-3:
                        total_loss = train_loss1 + train_loss2
                    else:
                        total_loss = train_loss1

                # apply gradients
                grad1 = m1_tape.gradient(total_loss, train_model1.trainable_variables)
                optimizer_1.apply_gradients(zip(grad1, train_model1.trainable_variables))
                grad2 = m2_tape.gradient(total_loss, train_model2.trainable_variables)
                optimizer_2.apply_gradients(zip(grad2, train_model2.trainable_variables))
                # early stopping
                if train_loss2 > 1e-3:
                    grad3 = m3_tape.gradient(total_loss, train_model3.trainable_variables)
                    optimizer_3.apply_gradients(zip(grad3, train_model3.trainable_variables))

                # add train loss of first batch
                if batch_idx == 0:
                    visualizer.add_train_loss1(train_loss1.numpy())
                    visualizer.add_train_loss2(train_loss2.numpy())
                batch_idx = batch_idx + 1

            # for each validation batch
            for data, label in validation_dataset:
                output_1 = train_model1(data)
                reconstruction = train_model2(output_1)
                classifier = train_model3(output_1)

                # calculate loss 1
                tmp = tf.cast(data[:, receptive_field:, :], dtype=tf.float32)
                mse = tf.square(tf.subtract(reconstruction, tmp))
                tmp3 = tf.reduce_mean(mse, axis=[1, 2])
                val_loss1 = tf.reduce_mean(tmp3)

                # calculate loss 2
                entropy_loss = tf.keras.losses.categorical_crossentropy(label, classifier)
                val_loss2 = tf.reduce_mean(entropy_loss)

                tmp_val_loss1.append(val_loss1.numpy())
                tmp_val_loss2.append(val_loss2.numpy())

            # add mean of validation loss for one batch
            visualizer.add_val_loss1(np.mean(np.array(tmp_val_loss1)))
            visualizer.add_val_loss2(np.mean(np.array(tmp_val_loss2)))

            end_time = time.time()
            epoch_time = end_time - start_time

            # print loss for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                print('Epoch {}, {} loss1: {:.4f} {:.4f}, loss2: {:.4f} {:.4f}, time: {:.1f}s'.format(epoch + 1,
                                                                                                      machine_type,
                                                                                                      visualizer.loss1_train[epoch],
                                                                                                      visualizer.loss1_val[epoch],
                                                                                                      visualizer.loss2_train[epoch],
                                                                                                      visualizer.loss2_val[epoch],
                                                                                                      epoch_time))

            # model save for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                # print("============== MODEL_SAVE ==============")
                model.model_save(param, machine_type, train_model1, train_model2, train_model3, epoch + 1)
                # print("============== FIGURE_SAVE ==============")
                history_img = "{root}/{model}/epoch_{epoch}/history_{machine}.png".format(root=param['model_root'],
                                                                                          model=param["model_dir"],
                                                                                          epoch=epoch + 1,
                                                                                          machine=machine_type)
                visualizer.save_figure(history_img)

                # # get mahalanobis mean and variance
                # t_dataset = np.concatenate([x for x, y in train_dataset], axis=0)
                # loss1 = []
                # loss2 = []
                # n_data = t_dataset.shape[0]
                #
                # for ii in tqdm(range(n_data)):
                #     t_data = t_dataset[ii:ii + 1, :, :]
                #     tmp = train_model1.predict(t_data)
                #     tmp1 = train_model2.predict(tmp)
                #     tmp2 = train_model3.predict(tmp)
                #
                #     data_loss1 = test.reconstruction_loss(machine_type, t_data, tmp1, dataset_mean)

if __name__ == '__main__':
    param_ = utils.yaml_load()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param_['gpu_num'])
    print(param_['model_dir'])
    main(param_)
