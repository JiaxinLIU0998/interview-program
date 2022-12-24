import numpy as np
from imputers.CSDI import CSDIImputer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



CSDI_model = CSDIImputer(3)

CSDI_model.train(
    #training_data,
    masking ='rm',
    missing_ratio_or_k = 0.2,
    trainset_path = './datasets/train_ptbxl_248.npy',
    valset_path = './datasets/val_ptbxl_248.npy',
    testset_path = './datasets/test_ptbxl_248.npy',
    epochs = 200,
    samples_generate = 10,
    path_save = "/home/root/jpm/SSSD/tf/upload_CSDI/results/train_ptbxl_248/CSDI/",
    batch_size = 8,
    lr = 1.0e-3,
    layers = 4,
    channels = 64,
    nheads = 8,
    difussion_embedding_dim = 128,
    beta_start = 0.0001,
    beta_end = 0.5,
    num_steps = 50,
    schedule = 'quad',
    is_unconditional = 0,
    timeemb = 128,
    featureemb = 16,
    target_strategy = 'random')

