# -*- coding: utf-8 -*-
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time

from utils.util import find_max_epoch, calc_diffusion_hyperparams
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from imputers.SSSDS4Imputer import SSSDS4Imputer


@tf.function
def train_step(X, y, net,loss_fn,optimizer):
    with tf.GradientTape() as tape:
        logits = net(X)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss_value.numpy()
    


def train(output_directory,
          #ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
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

    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    

    ## Custom data loading and reshaping ###
    training_data = np.load(trainset_config['train_data_path']) 
    training_data = training_data[4:,:,:]
    training_data = np.transpose(training_data,(0,2,1))
    training_data = np.split(training_data, 1090*4, 0) # 160,50,100,14
    training_data = np.array(training_data)

    #print('Data loaded, please check the shape ' + str(training_data.shape) +' is [number of data points, length, number of features]')
    #training_data = tf.convert_to_tensor(training_data)

    ### Create the tf.data ###
    ### I specifically use cpu to process this part to save gpu's memory, this may take some time, so I save the resulting file the first time runing it ###
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    count = 0
    for batch in training_data:
        if masking == 'rm':
            transposed_mask = get_mask_rm(batch[0], missing_k)
        elif masking == 'mnr':
            transposed_mask = get_mask_mnr(batch[0], missing_k)
        elif masking == 'bm':
            transposed_mask = get_mask_bm(batch[0], missing_k)

        mask = np.transpose(transposed_mask, [1,0]) 
        mask =  np.repeat(np.expand_dims(mask, axis=0), repeats=batch.shape[0],axis = 0)
        loss_mask = mask.astype(bool)
        batch = np.transpose(batch, [0,2,1]) 

        audio = batch
        cond = batch
        mask = mask
        loss_mask = loss_mask

        B, C, L = audio.shape  # B is batchsize, C is number of features, L is audio length
        diffusion_steps = np.random.randint(low = 0,high = T,size=(B, 1, 1)) # sample a diffusion step t for each of data point in the batch   
        z = np.random.normal(0, 1, audio.shape) 
        if only_generate_missing == 1: # mode D1: apply the diffusion process to the regions to be imputed only
            z = audio * mask + z * (1-mask)

        transformed_X = np.sqrt(Alpha_bar[diffusion_steps]) * audio + np.sqrt(1 - Alpha_bar[diffusion_steps]) * z  # generated x_t with noise
        if count == 0:
            all_transformed_X  = transformed_X
            all_cond = cond
            all_mask = mask
            all_diffusion_steps = np.reshape(diffusion_steps, [B, 1])
            all_z = z
        else:
            all_transformed_X = np.concatenate([all_transformed_X, transformed_X],axis=0)
            all_cond = np.concatenate([all_cond, cond],axis=0)
            all_mask = np.concatenate([all_mask, mask],axis=0)
            all_diffusion_steps = np.concatenate([all_diffusion_steps, np.reshape(diffusion_steps, [B, 1])],axis=0)
            all_z = np.concatenate([all_z, z],axis=0)

        count += 1

    np.save('all_transformed_X.npy',all_transformed_X)
    np.save('all_cond.npy',all_cond)
    np.save('all_mask.npy',all_mask)
    np.save('all_diffusion_steps.npy',all_diffusion_steps)
    np.save('all_z.npy',all_z)

    #all_transformed_X = np.load('all_transformed_X.npy')
    #all_cond = np.load('all_cond.npy')
    #all_mask = np.load('all_mask.npy')
    #all_diffusion_steps = np.load('all_diffusion_steps.npy')
    #all_z = np.load('all_z.npy')
    
    all_transformed_X = tf.convert_to_tensor(all_transformed_X,dtype=tf.float32)
    all_cond = tf.convert_to_tensor(all_cond,dtype=tf.float32)
    all_mask = tf.convert_to_tensor(all_mask,dtype=tf.float32)
    all_diffusion_steps = tf.convert_to_tensor(all_diffusion_steps,dtype=tf.float32)
    all_z = tf.convert_to_tensor(all_z,dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((tf.reshape(all_transformed_X,[-1,12,248]), tf.reshape(all_cond ,[-1,12,248])
                                                  ,tf.reshape(all_mask,[-1,12,248]),all_diffusion_steps),tf.reshape(all_z,[-1,12,248])))

    dataset = dataset.batch(8) # set the batch_size as 8. 
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    #training
    n_iter = 0
    iterator = iter(dataset)
    while n_iter < n_iters + 1:
        try:
            t1 = time.time()
            X,y = next(iterator) #X: (batch, batch, mask, loss mask(bool))
            loss = train_step(X,y,net,tf.keras.losses.MeanSquaredError(),optimizer)
            print(time.time()-t1)

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                net.write(output_directory +'./{}/{}.ckpt'.format(n_iter, n_iter),optimizer)
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

        except StopIteration:  
            print("current iterator done")
            iterator = iter(dataset)
                
   
    #y_set = y_set.prefetch(1)

    
    #net.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(),metrics = 'mse')
    
    #net.fit(x=dataset, steps_per_epoch=1,workers = 8,epochs=10)
    #print(net.summary())
    #print('My custom loss: ', net.loss_tracker.result().numpy())



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', type=str, default='./config/config_SSSDS4_ptx.json',
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
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)

    
