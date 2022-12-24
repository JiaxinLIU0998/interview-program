import os
import argparse
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, sampling, calc_diffusion_hyperparams
from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statistics import mean


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             masking,
             missing_k):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    missing_k (int)                   k missing time points for each channel across the length.
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
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    try: 
        net.load_weights(output_directory +'/{}/{}.ckpt'.format(ckpt_iter, ckpt_iter)).expect_partial()
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    batch_size, masking, missing_k = train_config['batch_size'],train_config['masking'],train_config['missing_k']
    testing_data = np.load(trainset_config['test_data_path'])
    testing_data = np.transpose(testing_data,(0,2,1))
    
    for count,single in enumerate(testing_data):
        repeated_sample = np.repeat(np.expand_dims(single,axis=0),num_samples,axis=0)
        if count == 0:
            repeated = repeated_sample
        else:
            repeated = np.concatenate([repeated,repeated_sample],axis=0)
            
    testing_data = tf.reshape(tf.convert_to_tensor(repeated),[-1,num_samples,repeated.shape[1],repeated.shape[2]])
    
    print('Data loaded')

    all_mse = []
    all_mae = []

    for i, batch in enumerate(testing_data):
        if masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)
        elif masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
            
        mask = tf.transpose(mask_T, [1,0]) 
        mask =  tf.repeat(tf.expand_dims(mask, axis=0), repeats=batch.shape[0],axis = 0)
        loss_mask = tf.cast(tf.where(mask==0,1,0),dtype=tf.bool) 
        batch = tf.transpose(batch, [0,2,1]) 
        
        sample_length = batch.shape[2]
        sample_channels = batch.shape[1]
        
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   z = batch,
                                   loss_mask = loss_mask)


        print('generated {} utterances of random_digit at iteration {} '.format(num_samples,ckpt_iter ))

        mse = mean_squared_error(generated_audio[loss_mask], batch[loss_mask])
        mae = mean_absolute_error(generated_audio[loss_mask], batch[loss_mask])
        all_mse.append(mse)
        all_mae.append(mae)
    
    print('Total MSE:', mean(all_mse))
    print('Total MAE:', mean(all_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_SSSDS4_bm_ptx.json',
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
             ckpt_iter=config["train_config"]['ckpt_iter'],
             num_samples=args.num_samples,
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"])

