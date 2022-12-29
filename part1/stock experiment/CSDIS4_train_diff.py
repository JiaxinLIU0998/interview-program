import numpy as np
from imputers.CSDIS4 import CSDIImputer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



CSDI_model = CSDIImputer()

CSDI_model.train(
    index = 'diff',
    trainset_path = './datasets/diff/diff_train.npy',
    valset_path = './datasets/diff/diff_val.npy',
    testset_path = './datasets/diff/diff_test.npy',
    mask_path =  './datasets/diff/diff_mask.npy',
    mask_test_path = './datasets/diff/diff_mask.npy',
    epochs = 500,
    samples_generate = 10,
    path_save =  './results/diff/CSDIS4/',
    batch_size = 16,
    lr = 1.0e-3,
    layers = 4,
    channels = 64,
    nheads = 8,
    diffussion_embedding_dim = 128,
    beta_start = 0.0001,
    beta_end = 0.5,
    num_steps = 50,
    schedule = 'quad',
    is_unconditional = 0,
    timeemb = 128,
    featureemb = 16,
    missing_ratio = 0.0,
    mean_std_path = './datasets/diff/diff_mean_std.pickle',
    lmax = 248,
    diff = True,)
