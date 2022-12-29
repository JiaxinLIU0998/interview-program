# -*- coding: utf-8 -*-
import os
import argparse
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import random
from utils.util import find_max_epoch, calc_diffusion_hyperparams
from imputers.SSSDS4Imputer import SSSDS4Imputer




def process_data(data_path,mask_path):
    """
    Process training data
    
    Paramutes:
    data_path:                            Dataset path
    mask_path:                            Mask path
    
    Returns:
    all_transformed_X (tf.tensor):        Noised input           
    all_cond (tf.tensor):                 Otiginal input
    all_mask (tf.tensor):                 Imputation mask
    all_diffusion_steps (tf.tensor):      Diffusion step     
    all_z (tf.tensor):                    Label 
    all_loss_mask (tf.tensor):            Imputation mask (bool), used for computing loss
    
    """

    
    ## Custom data loading and reshaping ###
    batch_size,missing_ratio = train_config['batch_size'],train_config['missing_ratio']
    data = np.load(data_path) 
    data = np.reshape(np.repeat(np.expand_dims(data,0),5,0),[-1,data.shape[1],data.shape[2]])
    drop_data = data.shape[0]%batch_size
    data = data[drop_data:,:,:]
    data = np.array(data)
    
    mask_template = np.load(mask_path) 

    _dh = diffusion_hyperparams

    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    count = 0
    count_sample = 1
    for sample in data:
        sample = np.nan_to_num(sample)
        time_stamp = sample.shape[0]
        num_of_tickers = sample.shape[1]
        mask_mvi = np.where(sample != 0, 1, 0)

        template_num = np.random.randint(low = 0,high = mask_template.shape[0],size = 1)[0]
        mask_imp = mask_template[template_num]

        #for i in range(mask_imp.shape[0]):
        if (1- mask_imp.sum()/(time_stamp*num_of_tickers))<missing_ratio:
            full_list = np.where(mask_imp[:,0]==1)[0]
            index = random.sample(list(full_list),k=int((mask_imp.sum()-time_stamp*num_of_tickers*(1-missing_ratio))//num_of_tickers))
            mask_imp[index,:] = 0
        
        mask_single = mask_imp * mask_mvi
        loss_mask_single = np.where(mask_mvi==0,False,True) * ~mask_single.astype(bool)
        
        mask_single = np.transpose(mask_single, [1,0]) 
        loss_mask_single = np.transpose(loss_mask_single, [1,0])
        sample = np.transpose(sample, [1,0]) 
        
        if count_sample == 1:
            mask  = mask_single
            batch = sample
            loss_mask = loss_mask_single
           
        else:
            mask = np.concatenate([mask, mask_single],axis=0)
            batch = np.concatenate([batch, sample],axis=0)
            loss_mask = np.concatenate([loss_mask, loss_mask_single],axis=0)
        
        if count_sample == batch_size:
          
            batch = np.reshape(batch,[batch_size,sample.shape[0],sample.shape[1]])
            mask = np.reshape(mask,[batch_size,sample.shape[0],sample.shape[1]])
            loss_mask = np.reshape(loss_mask,[batch_size,sample.shape[0],sample.shape[1]])
            
            assert batch.shape == mask.shape == loss_mask.shape

            audio = batch
            cond = batch

            B, C, L = audio.shape  # B is batchsize, C is number of features, L is audio length
            diffusion_steps = np.random.randint(low = 0,high = T,size=(B, 1, 1)) # sample a diffusion step t for each of data point in the batch   
            z = np.random.normal(0, 1, audio.shape) 
            z = audio * mask + z * (1-mask)

            transformed_X = np.sqrt(Alpha_bar[diffusion_steps]) * audio + np.sqrt(1 - Alpha_bar[diffusion_steps]) * z  # generated x_t with noise

            if count == 0:
                all_transformed_X  = transformed_X
                all_cond = cond
                all_mask = mask
                all_diffusion_steps = np.reshape(diffusion_steps, [B, 1])
                all_z = z
                all_loss_mask = loss_mask
                count += 1
            else:
                all_transformed_X = np.concatenate([all_transformed_X, transformed_X],axis=0)
                all_cond = np.concatenate([all_cond, cond],axis=0)
                all_mask = np.concatenate([all_mask, mask],axis=0)
                all_diffusion_steps = np.concatenate([all_diffusion_steps, np.reshape(diffusion_steps, [B, 1])],axis=0)
                all_z = np.concatenate([all_z, z],axis=0)
                all_loss_mask  = np.concatenate([all_loss_mask , loss_mask ],axis=0)
            
            count_sample = 1
        else:
            count_sample += 1
            

        
    return tf.convert_to_tensor(all_transformed_X,dtype=tf.float32), tf.convert_to_tensor(all_cond,dtype=tf.float32), tf.convert_to_tensor(all_mask,dtype=tf.float32),tf.convert_to_tensor(all_diffusion_steps,dtype=tf.float32), tf.convert_to_tensor(all_z,dtype=tf.float32), tf.convert_to_tensor(all_loss_mask,dtype=tf.bool)



def train(output_directory,
          epochs,
          learning_rate,
          batch_size,
         missing_ratio):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 1k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to compute validation loss, default is 150
    learning_rate (float):          learning rate
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
    net = SSSDS4Imputer(**model_config)
   
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    
    #generate model input and label
    train_transformed_X, train_cond, train_mask,train_diffusion_steps, train_z, train_loss_mask = process_data(trainset_config['train_data_path'],trainset_config['mask_path'])
    val_transformed_X, val_cond, val_mask, val_diffusion_steps, val_z, val_loss_mask = process_data(trainset_config['val_data_path'],trainset_config['mask_path'])
    
    dataset_train = (train_transformed_X, train_cond , train_mask , train_diffusion_steps, train_z, train_loss_mask)
    dataset_val = (val_transformed_X, val_cond , val_mask , val_diffusion_steps, val_z, val_loss_mask)
    
    # model compile
    net.compile(optimizer=optimizer)
    
    num_of_batch = (train_transformed_X.shape[0]//batch_size)
    
    checkpointer = [tf.keras.callbacks.ModelCheckpoint(output_directory +'/{epoch:06d}/{epoch:06d}.ckpt',monitor='val_loss', mode = "min", save_best_only=False, verbose=1,save_freq='epoch'), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)]
    
    val_freq = [(i+1) for i in range(epochs)]
    # model training
    
    
    net.fit(dataset_train,None,steps_per_epoch=num_of_batch,workers = 8,epochs=epochs,batch_size=batch_size,shuffle=True,
            callbacks=checkpointer,validation_data=(dataset_val,None),validation_batch_size=batch_size,validation_freq=val_freq)
    #print(net.summary())




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', type=str, default='./config/config_SSSDS4_HSI_final.json',
                        help='JSON file for configuration')
    parser.add_argument('--gpu', type=int, default=0,help='index of the GPU devide')
    
    args = parser.parse_args()
    args.gpu = 2
    
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
    model_config = config['wavenet_config']

    train(**train_config)

    
