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

from train import Visualizer


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
        train_dataset, validation_dataset = dataset.generate_dataset(param=param, machine=machine_type, dataset_dir=target_dir, mode=1)

        print("============== MODEL_TRAINING ==============")
        # set model parameters
        input_size = list(train_dataset)[0][0].shape[1]
        num_channel = param['model']['num_channel']
        num_classes = list(train_dataset)[0][1].shape[1]

        train_model3 = model.Resnet18(input_size, num_channel, 1, n_output=num_classes)

        # for each epoch
        for epoch in range(param['fit']['epochs']):
            # start time
            start_time = time.time()

            # initialize optimizers
            optimizer_3 = tf.keras.optimizers.Adam(param['learning_rate']['model3'], beta_1=0.85)

            # first batch index
            batch_idx = 0

            # initialize validation loss
            tmp_val_loss = []

            # for each train batch
            for data, label in train_dataset:
                with tf.GradientTape() as m3_tape:
                    classifier = train_model3(data)

                    # calculate loss 2
                    entropy_loss = tf.keras.losses.categorical_crossentropy(label, classifier)
                    train_loss = tf.reduce_mean(entropy_loss)

                    total_loss = train_loss

                # apply gradients
                grad3 = m3_tape.gradient(total_loss, train_model3.trainable_variables)
                optimizer_3.apply_gradients(zip(grad3, train_model3.trainable_variables))

                # add train loss of first batch
                if batch_idx == 0:
                    visualizer.add_train_loss2(train_loss.numpy())
                batch_idx = batch_idx + 1

            # for each validation batch
            for data, label in validation_dataset:
                classifier = train_model3(data)

                # calculate loss 2
                entropy_loss = tf.keras.losses.categorical_crossentropy(label, classifier)
                val_loss = tf.reduce_mean(entropy_loss)

                tmp_val_loss.append(val_loss.numpy())

            # add mean of validation loss for one batch
            visualizer.add_val_loss2(np.mean(np.array(tmp_val_loss)))

            end_time = time.time()
            epoch_time = end_time - start_time

            # print loss for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                print('Epoch {}, {} loss2: {:.4f} {:.4f}, time: {:.1f}s'.format(epoch + 1,
                                                                                machine_type,
                                                                                visualizer.loss2_train[epoch],
                                                                                visualizer.loss2_val[epoch],
                                                                                epoch_time))

            # model save for every 10 epochs
            if (epoch + 1) % param['fit']['save_epoch'] == 0:
                model.model_save(param, machine_type, 0, 0, train_model3, epoch + 1)
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
