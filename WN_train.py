import os
import glob
import sys
import numpy as np
import time
import tensorflow as tf

import utils
import dataset
import model

from train import Visualizer


def main(param):
    mode = utils.mode_check(param)
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(param['model_root'] + '/' + param['model_dir'], exist_ok=True)

    # load base_directory list
    data_dirs = utils.select_dirs(param, mode)

    # select training dataset
    dirs = utils.select_machine(param, data_dirs)

    # repeat for each machine directory
    for idx, target_dir in enumerate(dirs):
        print('\n===========================================================')
        print('[{idx}/{total}] {dirname}'.format(idx=idx + 1, total=len(dirs), dirname=target_dir))

        # initialize the visualizer
        visualizer = Visualizer()

        machine_type = os.path.split(target_dir)[1]

        visualizer.add_machine_type(machine_type)

        print("============== DATASET_GENERATOR ==============")
        train_dataset, validation_dataset = dataset.generate_dataset(param=param, machine=machine_type, dataset_dir=target_dir)

        print("============== MODEL_TRAINING ==============")
        input_size = list(train_dataset)[0][0].shape[1]
        num_channel = param['model']['num_channel']
        num_filters = param['model']['channel_multiply'] * num_channel
        kernel_size = param['model']['kernel_size']
        num_blocks = param['model']['num_blocks']

        train_model1 = model.Wavenet_blocks(input_size, num_channel, num_filters, kernel_size, num_blocks)
        train_model2 = model.Wavenet_reconstruction(input_size, num_channel, kernel_size, num_blocks)

        receptive_field = train_model2.receptive_field

        # for each epoch
        for epoch in range(param['fit']['epochs']):
            # start time
            start_time = time.time()

            # initialize optimizers
            optimizer_1 = tf.keras.optimizers.Adam(param['learning_rate']['model1'], beta_1=0.85)
            optimizer_2 = tf.keras.optimizers.Adam(param['learning_rate']['model2'], beta_1=0.85)

            # first batch index
            batch_idx = 0

            # initialize validation loss
            tmp_val_loss = []

            # for each train batch
            for data, label in train_dataset:
                with tf.GradientTape() as m1_tape, tf.GradientTape() as m2_tape:
                    skip_connection = train_model1(data)
                    reconstruction = train_model2(skip_connection)

                    # calculate loss
                    tmp = tf.cast(data[:, receptive_field:, :], dtype=tf.float32)
                    mse = tf.square(tf.subtract(reconstruction, tmp))
                    tmp3 = tf.reduce_mean(mse, axis=[1, 2])
                    train_loss = tf.reduce_mean(tmp3)

                    # calculate total loss
                    total_loss = train_loss

                # apply gradients
                grad1 = m1_tape.gradient(total_loss, train_model1.trainable_variables)
                optimizer_1.apply_gradients(zip(grad1, train_model1.trainable_variables))

                grad2 = m2_tape.gradient(total_loss, train_model2.trainable_variables)
                optimizer_2.apply_gradients(zip(grad2, train_model2.trainable_variables))

                if batch_idx == 0:
                    visualizer.add_train_loss1(train_loss.numpy())
                batch_idx = batch_idx + 1

            # for each validation batch
            for data, label in validation_dataset:
                skip_connection = train_model1(data)
                reconstruction = train_model2(skip_connection)

                # calculate loss
                tmp = tf.cast(data[:, receptive_field:, :], dtype=tf.float32)
                mse = tf.square(tf.subtract(reconstruction, tmp))
                tmp3 = tf.reduce_mean(mse, axis=[1, 2])
                val_loss = tf.reduce_mean(tmp3)

                tmp_val_loss.append(val_loss.numpy())

            # add mean of validation loss for one batch
            visualizer.add_val_loss1(np.mean(np.array(tmp_val_loss)))

            end_time = time.time()
            epoch_time = end_time - start_time

            # print loss for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                print('Epoch {}, {} loss1: {:.4f} {:.4f}, time: {:.1f}s'.format(epoch + 1,
                                                                                machine_type,
                                                                                visualizer.loss1_train[epoch],
                                                                                visualizer.loss1_val[epoch],
                                                                                epoch_time))

            # model save for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                model.model_save(param, machine_type, train_model1, train_model2, 0, epoch + 1)
                history_img = "{root}/{model}/epoch_{epoch}/history_{machine}.png".format(root=param['model_root'],
                                                                                          model=param["model_dir"],
                                                                                          epoch=epoch + 1,
                                                                                          machine=machine_type)
                visualizer.save_figure(history_img)


if __name__ == '__main__':
    param_ = utils.yaml_load()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param_['gpu_num'])
    print(param_['model_dir'])
    main(param_)
