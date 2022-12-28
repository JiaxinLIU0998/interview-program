# -*- coding: utf-8 -*-
import os
import argparse
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.util import find_max_epoch, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from imputers.SSSDSAImputer import SSSDSAImputer




def process_data():
    """
    Process training data
    
    Returns:
    all_transformed_X (tf.tensor):        Noised input           
    all_cond (tf.tensor):                 Otiginal input
    all_mask (tf.tensor):                 Imputation mask
    all_diffusion_steps (tf.tensor):      Diffusion step     
    all_z (tf.tensor):                    Label 
    all_loss_mask (tf.tensor):            Imputation mask (bool), used for computing loss
    
    """

    
    ## Custom data loading and reshaping ###
    batch_size, masking, missing_k = train_config['batch_size'],train_config['masking'],train_config['missing_k']
    training_data = np.load(trainset_config['train_data_path']) 
    drop_data = training_data.shape[0]%batch_size
    training_data = training_data[drop_data:,:,:]
    training_data = np.transpose(training_data,(0,2,1))
    training_data = np.split(training_data, training_data.shape[0]//batch_size, 0) # total batch: 8720//2
    training_data = np.array(training_data)
    training_data = tf.convert_to_tensor(training_data)

    _dh = diffusion_hyperparams

    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    Alpha_bar = tf.convert_to_tensor(Alpha_bar,dtype = tf.float32)
    count = 0
    for batch in training_data:
        if masking == 'rm':
            transposed_mask = get_mask_rm(batch[0], missing_k)
        elif masking == 'mnr':
            transposed_mask = get_mask_mnr(batch[0], missing_k)
        elif masking == 'bm':
            transposed_mask = get_mask_bm(batch[0], missing_k)

        mask = tf.transpose(transposed_mask, [1,0]) 
        mask =  tf.repeat(tf.expand_dims(mask, axis=0), repeats=batch.shape[0],axis = 0)
        loss_mask = tf.cast(tf.where(mask==0,1,0),dtype=tf.bool) 
        batch = tf.transpose(batch, [0,2,1]) 

        assert batch.shape == mask.shape == loss_mask.shape

        audio = batch
        cond = batch

        B, C, L = audio.shape  # B is batchsize, C is number of features, L is audio length
        
        diffusion_steps = tf.experimental.numpy.random.randint(low = 0,high = T,size=(B, 1, 1)) # sample a diffusion step t for each of data point in the batch
        z = tf.random.normal(shape = audio.shape,mean = 0,stddev=1) 
        z = audio * tf.cast(mask,dtype=tf.float32) + z * tf.cast(1-mask,dtype=tf.float32)
        transformed_X = tf.math.sqrt(tf.gather(Alpha_bar,diffusion_steps,axis = 0)) * audio + tf.math.sqrt(1 - tf.gather(Alpha_bar,diffusion_steps,axis = 0)) * z  # generated x_t with noise


        if count == 0:
            all_transformed_X  = transformed_X
            all_cond = cond
            all_mask = mask
            all_diffusion_steps = tf.reshape(diffusion_steps, [B, 1])
            all_z = z
            all_loss_mask = loss_mask
        else:
            all_transformed_X = tf.concat([all_transformed_X, transformed_X],axis=0)
            all_cond = tf.concat([all_cond, cond],axis=0)
            all_mask = tf.concat([all_mask, mask],axis=0)
            all_diffusion_steps = tf.concat([all_diffusion_steps, tf.reshape(diffusion_steps, [B, 1])],axis=0)
            all_z = tf.concat([all_z, z],axis=0)
            all_loss_mask  = tf.concat([all_loss_mask,loss_mask],axis=0)

        count += 1

    return all_transformed_X, all_cond, all_mask, all_diffusion_steps, all_z, all_loss_mask



def train(output_directory,
          ckpt_iter,
          iters_per_ckpt,
          n_iters,
          learning_rate,
          masking,
          missing_k,
          batch_size):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    learning_rate (float):          learning rate
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    batch_size (int):               traini config: batch size.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)
    
   
    # predefine model
    net = SSSDSAImputer(**model_config)
   
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    
    #generate model input and label
    all_transformed_X, all_cond, all_mask, all_diffusion_steps, all_z, all_loss_mask = process_data()
    dataset_x = (all_transformed_X, all_cond , all_mask , all_diffusion_steps, all_z, all_loss_mask)
    
    # model compile
    net.compile(optimizer=optimizer)
    
    checkpointer = [tf.keras.callbacks.ModelCheckpoint(output_directory +'/{epoch:06d}/{epoch:06d}.ckpt',monitor='train_loss',
                                                       mode = "min", save_best_only=False, verbose=1,period=iters_per_ckpt)]
    
    # model training
    net.fit(dataset_x,None,steps_per_epoch=1,workers = 8,epochs=n_iters,batch_size=batch_size,shuffle=False,callbacks=checkpointer)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', type=str, default='./config/config_SSSDSA_ptx.json',
                        help='JSON file for configuration')
    parser.add_argument('--gpu', type=int, default=0,help='index of the GPU devide')
    
    args = parser.parse_args()
    args.gpu = 0
    
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu],'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], enable=True)
   
   
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters
    
    global model_config
    model_config = config['sashimi_config']

    train(**train_config)

    
