import glob
import argparse
import sys
import os
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
import tensorflow as tf
import logging
import itertools
import re

from tqdm import tqdm


def yaml_load():
    with open('params.yaml') as stream:
        param = yaml.safe_load(stream)
    return param


def mode_check(param):
    """
    check 'development' mode or 'evaluation' mode
    :param param: parameter dictionary
    :return: dev: True / eval: False
    """
    if param['mode'] == 'dev':
        return True
    elif param['mode'] == 'eval':
        return False
    else:
        return None


def select_dirs(param,
                mode):
    """
    get dataset directory
    :param param: parameter dictionary
    :param mode: dev: True / eval: False
    :return: dataset directory
    """
    if mode:
        print('load_directory <- development')
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_dir"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        print('load_directory <- evaluation')
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_dir"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs


def select_machine(param,
                   dirs):
    """
    select training machine
    :param param: parameter dictionary
    :param dirs: machine dataset directory
    :return: selected machine dataset directory
    """
    output = []
    if param['dataset']['toycar']:
        output.append(dirs[0])
    if param['dataset']['toyconveyor']:
        output.append(dirs[1])
    if param['dataset']['fan']:
        output.append(dirs[2])
    if param['dataset']['pump']:
        output.append(dirs[3])
    if param['dataset']['slider']:
        output.append(dirs[4])
    if param['dataset']['valve']:
        output.append(dirs[5])

    return output


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    _dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(_dir_path))
    # extract id
    _machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return _machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             mode,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    print("target_dir : {}".format(target_dir + "_" + id_name))
    # com.logger.info("target_dir : {}".format(target_dir + "_" + id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        print("test_file  num : {num}".format(num=len(files)))
        # com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
            # com.logger.exception("no_wav_file!!")
        # print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        print("test_file  num : {num}".format(num=len(files)))
        # com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
            # com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels


def calculate_mahalanobis_distance(machine_str):

    return