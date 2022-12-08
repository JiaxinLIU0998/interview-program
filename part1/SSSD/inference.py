import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import json
import numpy as np
import tensorflow as tf

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, sampling, calc_diffusion_hyperparams

from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from statistics import mean


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             masking,
             missing_k,
             only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
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
    if use_model == 0:
        net = DiffWaveImputer(**model_config)
    elif use_model == 1:
        net = SSSDSAImputer(**model_config)
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config)
    else:
        print('Model chosen not available.')
    

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    try:
        net.restore(output_directory +'/{}/{}.ckpt'.format(ckpt_iter, ckpt_iter)).expect_partial()
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    
    testing_data = np.load(trainset_config['test_data_path'])
    testing_data = testing_data[312:,::]
    testing_data = np.transpose(testing_data,(0,2,1))
    testing_data = np.split(testing_data, 17, 0) # 160,50,100,14
    testing_data = np.array(testing_data)
    
    print('Data loaded')

    all_mse = []

    
    for i, batch in enumerate(testing_data):
        if masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)
        elif masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
        
        mask = np.transpose(mask_T, [1,0]) 
        mask = np.repeat(np.expand_dims(mask, axis=0), repeats=batch.shape[0],axis = 0)
    
        batch = np.transpose(batch, [0,2,1]) 
        
        sample_length = batch.shape[2]
        sample_channels = batch.shape[1]
        
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)


        print('generated {} utterances of random_digit at iteration {} '.format(num_samples,ckpt_iter, ))

        bool_mask = mask.astype(bool) 
        mse = mean_squared_error(generated_audio[bool_mask], batch[bool_mask])
        all_mse.append(mse)
    
    print('Total MSE:', mean(all_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config/config_SSSDS4_ptx.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
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
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             use_model=train_config["use_model"],
             data_path=trainset_config["test_data_path"],
             masking=train_config["masking"],
             missing_k=train_config["missing_k"],
             only_generate_missing=train_config["only_generate_missing"])
