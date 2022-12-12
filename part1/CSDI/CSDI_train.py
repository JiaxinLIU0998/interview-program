import numpy as np
from imputers.CSDI import CSDIImputer



CSDI_model = CSDIImputer(1)

CSDI_model.train(
    #training_data,
    masking ='rm',
    missing_ratio_or_k = 0.2,
    trainset_path = './datasets/train_ptbxl_248.npy',
    testset_path = './datasets/test_ptbxl_248.npy',
    epochs = 200,
    samples_generate = 10,
    path_save = "./results/train_ptbxl_248/CSDI",
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