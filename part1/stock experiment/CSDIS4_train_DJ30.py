import numpy as np
from imputers.CSDIS4 import CSDIImputer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



CSDI_model = CSDIImputer()

CSDI_model.train(
    index = 'DJ30',
    trainset_path = './datasets/DJ30/DJ30_train.npy',
    valset_path = './datasets/DJ30/DJ30_val.npy',
    testset_path = './datasets/DJ30/DJ30_test.npy',
    mask_path =  './datasets/DJ30/DJ30_mask_train.npy',
    mask_test_path = './datasets/DJ30/DJ30_mask_test.npy',
    epochs = 500,
    samples_generate = 10,
    path_save =  './results/DJ30/CSDIS4/',
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
    missing_ratio = 0.3,
    mean_std_path = './datasets/DJ30/DJ30_mean_std.pickle',
    lmax = 248,)
