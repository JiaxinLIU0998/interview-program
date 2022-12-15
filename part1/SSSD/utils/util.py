# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import random


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        try:
            epoch = max(epoch, int(f))
        except:
            continue
    return epoch



# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """
    return np.random.normal(0,1,size)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = tf.math.log(tf.convert_to_tensor(10000.0)) / (half_dim - 1)
    
    _embed = tf.math.exp(tf.cast(tf.experimental.numpy.arange(start = 0,stop = half_dim),dtype = tf.float32) * -_embed)
    _embed = tf.cast(diffusion_steps,dtype=tf.float32) * _embed
    diffusion_step_embed = tf.concat((tf.math.sin(_embed),
                                      tf.math.cos(_embed)), 1)
    
    assert diffusion_step_embed.shape[0] == diffusion_steps.shape[0]
    assert diffusion_step_embed.shape[1] == diffusion_step_embed_dim_in

    return diffusion_step_embed



def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (shape=(T, ))
    """

    Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] = Alpha_bar[t-1]*Alpha_bar[t]
        Beta_tilde[t] = Beta_tilde[t] * (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])
       
    Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask,z,loss_mask):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) , shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)
            

    for t in range(T - 1, -1, -1):
        x = x * tf.cast((1 - mask),dtype=tf.float32) + cond * tf.cast((mask),dtype=tf.float32)
        diffusion_steps = (t * tf.ones((size[0], 1))) # use the corresponding reverse step

        epsilon_theta = net.predict((x, cond, mask, diffusion_steps, z,loss_mask),batch_size=x.shape[0])  # predict \epsilon according to \epsilon_\theta
        # update x_{t-1} to \mu_\theta(x_t)
        x = (x - (1 - Alpha[t]) / np.sqrt(1 - Alpha_bar[t]) * epsilon_theta.numpy()) / np.sqrt(Alpha[t])
        if t > 0:
            x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
    
    assert x.shape == size

    return x




def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    mask = tf.ones(sample.shape)
    mask = tf.Variable(mask,trainable=False)
    length_index = tf.convert_to_tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = tf.random.shuffle([i for i in range(len(length_index))])
        idx = perm[0:k]
        tensor=tf.tensor_scatter_nd_update(mask[:,channel],tf.expand_dims(idx,axis = 1),[0]*len(idx))
        mask[:,channel].assign(tensor)
    mask = tf.convert_to_tensor(mask)
    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = tf.ones(sample.shape)
    length_index = tf.convert_to_tensor(range(mask.shape[0]))
    list_of_segments_index = tf.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = tf.ones(sample.shape)
    length_index = tf.convert_to_tensor(range(mask.shape[0]))
    list_of_segments_index = tf.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask
