import os
import argparse
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pickle

from utils.util import find_max_epoch, sampling, calc_diffusion_hyperparams
from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statistics import mean


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             mask_path):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
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
    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    ckpt_iter = find_max_epoch(ckpt_path)
    try: 
        net.load_weights(output_directory +'/{:06d}/{:06d}.ckpt'.format(ckpt_iter, ckpt_iter)).expect_partial()
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    #batch_size, masking, missing_k = train_config['batch_size'],train_config['masking'],train_config['missing_k']
    testing_data = np.load(data_path)
    mask = np.load(mask_path) 
    mask = np.repeat(np.expand_dims(mask,axis=0),num_samples,axis=0)
    loss_mask = ~mask.astype(bool)
    mask = np.transpose(mask, [0,2,1]) 
    loss_mask = np.transpose(loss_mask, [0,2,1])

    with open(trainset_config['mean_std'], 'rb') as handle:
        mean_std = pickle.load(handle)
    
    scaler = []
    for i in mean_std.keys():
        scaler.append(mean_std[i][1])
    scaler = np.array(scaler)
    scaler = np.repeat(np.expand_dims(scaler,axis=1),testing_data.shape[1],axis=1)
    
    testing_data = np.repeat(np.expand_dims(testing_data,axis=1),num_samples,axis=1)
    
  
    print('Data loaded')

    all_mse = []
    all_mae = []
    

    for i, batch in enumerate(testing_data):
        
        batch = tf.transpose(batch, [0,2,1]) 
        
        sample_length = batch.shape[2]
        sample_channels = batch.shape[1]
        
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=tf.cast(batch,dtype=tf.float32),
                                   mask=tf.cast(mask,dtype=tf.float32),
                                   z = tf.cast(batch,dtype=tf.float32),
                                   loss_mask = loss_mask)


        print('generated {} utterances of random_digit at iteration {} '.format(num_samples,ckpt_iter ))
        
        
        mse = tf.reduce_sum((((tf.cast(generated_audio,dtype=tf.float32) - tf.cast(batch,dtype=tf.float32)) * tf.where(loss_mask,1.0,0.0)) ** 2) * (scaler ** 2)).numpy()/tf.reduce_sum(tf.where(loss_mask,1,0)).numpy()
        mae = tf.reduce_sum((tf.math.abs(((tf.cast(generated_audio,dtype=tf.float32) - tf.cast(batch,dtype=tf.float32)) * tf.where(loss_mask,1.0,0.0)))* scaler)).numpy()/tf.reduce_sum(tf.where(loss_mask,1,0)).numpy()



        #mse = mean_squared_error(generated_audio[loss_mask], batch[loss_mask])
        #mae = mean_absolute_error(generated_audio[loss_mask], batch[loss_mask])
        print(mse)
        print(mae)
        all_mse.append(mse)
        all_mae.append(mae)
        print(i)
    
    print('Total MSE:', mean(all_mse))
    print('Total MAE:', mean(all_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_SSSDS4_diff.json',
                        help='JSON file for configuration')
    parser.add_argument('-n', '--num_samples', type=int, default=10,
                        help='Number of utterances to be generated')
    args = parser.parse_args()
    
    gpus = tf.config.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[0],'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    gen_config = config['gen_config']

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

    generate(**gen_config,
             num_samples=args.num_samples,
             data_path=trainset_config["test_data_path"],
             mask_path=trainset_config['mask_path'])
