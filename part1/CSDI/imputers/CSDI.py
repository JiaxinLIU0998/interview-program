import numpy as np
import random
import pickle
import math
import argparse
import datetime
import json
import os
from functools import partial
from imputers.transformerencoder import EncoderLayer as transformerencoder
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



''' Standalone CSDI imputer. The imputer class is located in the last part of the notebook, please see more documentation there'''


def MultiStepLR(initial_learning_rate, lr_steps, lr_rate, name='MultiStepLR'):
    """Multi-steps learning rate scheduler."""
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr_steps_value)


def train(model, config, train_loader, val_loader, valid_epoch_interval=1, path_save="",batch_size=16):
    num_of_batch = (train_loader[0].shape[0]//batch_size)
    
    p1 = int(0.75 * config["epochs"] * num_of_batch)
    p2 = int(0.9 * config["epochs"] * num_of_batch)
    
    
    lr_scheduler = MultiStepLR(config["lr"], [p1, p2], 0.1)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,epsilon=1e-6)
    
    model.compile(optimizer=optimizer)
    
    checkpointer = [tf.keras.callbacks.ModelCheckpoint(path_save +'{epoch:06d}/{epoch:06d}.ckpt',monitor='val_loss',
                                                       mode = "min", save_best_only=False, verbose=1,save_freq='epoch')]
    
    # model training
    val_freq = [(i+1)*valid_epoch_interval for i in range(config["epochs"]//valid_epoch_interval)]
    
    model.fit(x=train_loader, y=None,batch_size=batch_size, steps_per_epoch=(train_loader[0].shape[0]//batch_size),
              epochs=config["epochs"],shuffle=False,callbacks=checkpointer,validation_data=(val_loader,None),
              validation_batch_size=batch_size,validation_freq=val_freq)

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    forecast = tf.cast(forecast,tf.float32)
    target = tf.cast(target,tf.float32)
    return  2 * tf.math.reduce_sum(tf.math.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)), axis=0)



def calc_denominator(target, eval_points):
    return tf.math.reduce_sum(tf.math.abs(target * eval_points), axis=0)


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(tf.convert_to_tensor(np.quantile(forecast[j: j + 1], quantiles[i], axis=1)))
        q_pred = tf.concat(q_pred,axis = 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    return CRPS.numpy() / len(quantiles)



def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save="",batch_size=16):
    #model.load_weights('/home/root/jpm/SSSD/tf/upload_CSDI/results/train_ptbxl_248/CSDI/000001/000001.ckpt')
 
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []
    
    test_loader = tf.data.Dataset.from_tensor_slices(test_loader)
    test_loader = test_loader.batch(batch_size)
 
    iterator = iter(test_loader)
        
    batch_no = 0
    while 1:
        try:
            test_batch = next(iterator)
            output = model.predict_test(test_batch, nsample)

            samples, c_target, eval_points, observed_points, observed_time = output

            samples = tf.transpose(samples, perm = [0, 1, 3, 2])
            c_target = tf.transpose(c_target, perm = [0, 2, 1])
            eval_points = tf.transpose(eval_points, perm = [0, 2, 1])
            observed_points = tf.transpose(observed_points, perm = [0, 2, 1])

            samples_median = tfp.stats.percentile(samples, 50.0, interpolation='midpoint',axis=1)
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            #samples_median = tf.sparse.from_dense(samples_median)
           
            mse_current = (((tf.cast(samples_median, dtype=tf.float32) - c_target) * eval_points) ** 2) * (scaler ** 2)
            mae_current = (tf.math.abs((tf.cast(samples_median, dtype=tf.float32) - c_target) * eval_points))* scaler

            mse_total += tf.reduce_sum(mse_current).numpy()
            mae_total += tf.reduce_sum(mae_current).numpy()
            evalpoints_total += tf.reduce_sum(eval_points).numpy()
           
            print(mse_total)
            print(evalpoints_total)
            batch_no += 1

        except StopIteration:  
            print("iterator done")
            break;

    with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
        
        all_target = tf.concat(all_target, axis=0)
        all_evalpoint = tf.concat(all_evalpoint, axis=0)
        all_observed_point = tf.concat(all_observed_point, axis=0)
        all_observed_time = tf.concat(all_observed_time, axis=0)
        all_generated_samples = tf.concat(all_generated_samples, axis=0)

        pickle.dump(
            [
                all_generated_samples,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                scaler,
                mean_scaler,
            ],
            f,
        )

    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

    with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
        pickle.dump(
            [
                np.sqrt(mse_total / evalpoints_total),
                mae_total / evalpoints_total, 
                CRPS
            ], 
            f)
        print("RMSE:", np.sqrt(mse_total / evalpoints_total))
        print("MAE:", mae_total / evalpoints_total)
        print("CRPS:", CRPS)


    return all_generated_samples.numpy()




class Conv1d_with_init(tf.keras.layers.Layer):
    """Implementation of convolution 1D over the feature dimention"""
    
    def __init__(self, in_channels, out_channels, kernel_size, init=None):
        """Initializer.
        Args:
            in_channels: int, input channels.
            out_channels: int, output channels.
            kernel_size: int, size of the kernel.
            dilation: int, dilation rate.
        """
        super(Conv1d_with_init, self).__init__()
        if init == None:
            init = tf.keras.initializers.HeNormal()
        
        self.kernel = tf.Variable(init([kernel_size, in_channels, out_channels], dtype=tf.float32),trainable=True)
        self.bias = tf.Variable(tf.zeros([1, 1, out_channels], dtype=tf.float32),trainable=True)

    def call(self, x):
        """Pass to convolution 1d.
        Args:
            inputs: tf.Tensor, [B, K, L], input tensor.
        Returns:
            outputs: tf.Tensor, [B, C, L], output tensor.
        """
        out = tf.transpose(tf.nn.conv1d(tf.transpose(x,(0,2,1)), self.kernel, 1, padding='SAME', dilations=1) + self.bias,(0,2,1))

        return out


class DiffusionEmbedding(tf.keras.Model):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
            
        self.embedding  = tf.Variable(self._build_embedding(num_steps, embedding_dim / 2),trainable=False,name = "embedding",shape=self._build_embedding(num_steps, embedding_dim / 2).shape)
      
        self.projection1 = tf.keras.layers.Dense(units = embedding_dim,activation=None)
        self.projection2 = tf.keras.layers.Dense(units = projection_dim,activation=None)
   
    
    def call(self, diffusion_step):
        x = tf.gather(self.embedding, axis=0, indices=diffusion_step)
       
        x = self.projection1(x)
        x = tf.nn.silu(x)
      
        x = self.projection2(x)
        x = tf.nn.silu(x)
      
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.expand_dims(tf.range(start=0, limit=num_steps, delta=1),-1)
        frequencies = 10.0 ** (tf.expand_dims(tf.range(start=0, limit=dim, delta=1)/(dim - 1) * 4.0,0))
       
        table = tf.cast(steps,dtype = tf.float32) * frequencies  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis = 1)
    
        return table


class diff_CSDI(tf.keras.Model):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1,init = tf.keras.initializers.Constant(0))
        
        
        self.residual_layers = [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        

    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        
        x = tf.reshape(x,(B, inputdim, K * L))
      
        x = self.input_projection(x)
        x = tf.nn.relu(x)
        
        x = tf.reshape(x,[B, self.channels, K, L])
       

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
        
        x = tf.math.reduce_sum(tf.stack(skip),axis = 0)/ math.sqrt(len(self.residual_layers))
        x = tf.reshape(x,[B, self.channels, K*L])
     
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = tf.nn.relu(x)
     
        
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.reshape(x,[B, K, L])
        return x



class ResidualBlock(tf.keras.Model):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        
        self.diffusion_projection = tf.keras.layers.Dense(channels)
        
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.time_layer = transformerencoder(h=nheads, d_k=64, d_v=64, d_model=64, d_ff=2048, rate=0)
        self.feature_layer = transformerencoder(h=nheads, d_k=64, d_v=64, d_model=64, d_ff=2048, rate=0)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
       
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]),perm = [0, 2, 1, 3]),[B * K, channel, L])
        y = tf.transpose(self.time_layer(tf.transpose(y,perm = [2, 0, 1])),perm = [1, 2, 0])
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, K, channel, L]),perm = [0, 2, 1, 3]),[B, channel, K * L])
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
      
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]),perm = [0, 3, 1, 2]),[B * L, channel, K])
        y = tf.transpose(self.feature_layer(tf.transpose(y,perm = [2, 0, 1])),perm = [1, 2, 0])
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, L, channel, K]),perm = [0, 2, 3, 1]),[B, channel, K * L])
      
        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        x = tf.reshape(x,(B, channel, K * L))
        base_shape = [B, channel, K, L]
       
        diffusion_emb = self.diffusion_projection(diffusion_emb)#.unsqueeze(-1)  # (B,channel,1)
        y = x + tf.expand_dims(diffusion_emb,axis = -1)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        
       
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        
        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info,[B, cond_dim, K * L])
       
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        y = y + cond_info

        gate, filter = tf.split(y, 2,axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)

        y = self.output_projection(y)
        residual, skip = tf.split(y, 2, axis=1)
        
        x = tf.reshape(x,base_shape)
        residual = tf.reshape(residual,base_shape)
        skip = tf.reshape(skip,base_shape)
        return (x + residual) / math.sqrt(2.0), skip




class CSDI_base(tf.keras.Model):
    def __init__(self, config,target_dim):
        
        super().__init__()
        
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
            
        self.embed_layer = tf.keras.layers.Embedding(self.target_dim, self.emb_feature_dim)
       
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        
        
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        
        self.alpha_torch = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(self.alpha,dtype = tf.float32),1),1)
     
    def time_embedding(self, pos, d_model=128):
        
        pe = tf.zeros([tf.shape(pos)[0], tf.shape(pos)[1], d_model],dtype = tf.float32)
        
        concat = tf.zeros([tf.shape(pos)[0], tf.shape(pos)[1], d_model//2],dtype = tf.float32)
        
        
        position = tf.expand_dims(pos,axis=2)
       
        
        div_term = 1 / tf.math.pow(10000.0, tf.cast(tf.range(start = 0, limit = d_model, delta = 2) / d_model,dtype=tf.float32))
                
        part1 = tf.concat([tf.math.sin(tf.cast(position,dtype = tf.float32) * div_term),concat],axis=2)
        part2 = tf.concat([concat,tf.math.cos(tf.cast(position,dtype = tf.float32) * div_term)],axis=2)
   
        pe = pe + part1 + part2
    
        return pe
    
    
    def get_side_info(self, observed_tp, cond_mask):
        
        B, K, L = cond_mask.shape
     
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        
        time_embed = tf.broadcast_to(tf.expand_dims(time_embed,axis = 2), [B, L, K, time_embed.shape[2]])

        feature_embed = self.embed_layer(tf.range(start=0, limit=self.target_dim, delta=1))  # (K,emb)
      
        feature_embed = tf.broadcast_to(tf.expand_dims(tf.expand_dims(feature_embed,axis=0),axis=0),[B, L, feature_embed.shape[0], feature_embed.shape[1]])
       
        
        side_info = tf.concat([time_embed, feature_embed], axis=-1)  # (B,L,K,*)
        
        side_info = tf.transpose(side_info,perm = [0, 3, 2, 1])  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask,axis=1)  # (B,1,K,L)
            side_info = tf.concat([side_info, side_mask], axis=1)

        return side_info


    
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info,set_t=-1):
        B, K, L = observed_data.shape
        t = tf.random.uniform(shape=[B], minval=0, maxval=self.num_steps, dtype=tf.int32)
       
        current_alpha = tf.gather(self.alpha_torch, axis=0, indices=t)

        noise = tf.random.normal(shape=[B,observed_data.shape[1],observed_data.shape[2]], mean=0, stddev=1, dtype=tf.float32)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        
        num_eval = tf.reduce_sum(target_mask)
       
        loss = tf.reduce_sum(residual ** 2)/ num_eval 
     
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data,axis=1)
        else:
            
            cond_obs = tf.expand_dims((cond_mask * observed_data),1)
            noisy_target = tf.expand_dims(((1 - cond_mask) * noisy_data),1)
            total_input = tf.concat([cond_obs, noisy_target], axis=1) 
         
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        #imputed_samples = tf.zeros([B, n_samples, K, L])
        imputed_samples = np.zeros([B, n_samples, K, L])
    
        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = tf.random.random(shape=noisy_obs.shape, minval=0, maxval=1, dtype=tf.float32)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            
            current_sample = tf.random.random(shape=observed_data.shape, minval=0, maxval=1, dtype=tf.float32)
           
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input,axis=1)
                   
                else:
                    cond_obs = tf.expand_dims((cond_mask * observed_data),axis=1)
                    noisy_target = tf.expand_dims(((1 - cond_mask) * current_sample),axis=1)
                    diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                  
                predicted = self.diffmodel(diff_input, side_info, tf.convert_to_tensor([t]))
            
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = tf.random.random(shape=current_sample.shape, minval=0, maxval=1, dtype=tf.float32)
                   
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            
            #print(current_sample.numpy())
            imputed_samples[:, i] = current_sample.numpy()
            
        return tf.convert_to_tensor(imputed_samples)
    
    
    def process_data(self, batch):
       
        observed_data = batch[0]
        observed_mask = batch[1]
        observed_tp = batch[3]
        gt_mask = batch[2]
       
        
        observed_data = tf.transpose(observed_data,perm = [0,2,1])
        observed_mask = tf.transpose(observed_mask,perm = [0,2,1])
        gt_mask = tf.transpose(gt_mask,perm = [0,2,1])
        
        cut_length = tf.zeros(observed_data.shape[0],dtype=tf.int64)
        for_pattern_mask = observed_mask

        return (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,cut_length)

    
    def call(self, batch):
        (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,_) = self.process_data(batch)
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss 
        
        self.add_loss(loss_func(observed_data, cond_mask, observed_mask, side_info))

        return loss_func(observed_data, cond_mask, observed_mask, side_info)

    def predict_test(self, batch, n_samples):
        (observed_data,observed_mask,observed_tp,gt_mask,_,cut_length) = self.process_data(batch)
       
        cond_mask = gt_mask
        target_mask = (observed_mask - cond_mask).numpy()
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        for i in range(len(cut_length)):  
            print(target_mask[i, ..., 0: cut_length[i].numpy()])
            target_mask[i, ..., 0: cut_length[i].numpy()] = 0
                
        return samples, observed_data, tf.convert_to_tensor(target_mask), observed_mask, observed_tp

    


def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_masks = ~tf.math.is_nan(data)
    masks = tf.reshape(observed_masks,[-1])
    obs_indices = tf.reshape(tf.where(masks),[-1]).numpy().tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    gt_masks = tf.convert_to_tensor([True if (i not in miss_indices and masks[i]) else False for i in range(masks.shape[0])],dtype=tf.bool)
    gt_masks = tf.reshape(gt_masks,observed_masks.shape)
    observed_values = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    observed_masks = tf.cast(observed_masks,dtype = tf.float32)
    gt_masks = tf.cast(gt_masks,dtype = tf.float32)

    return observed_values, observed_masks, gt_masks


def mask_missing_train_nrm(data, k_segments=5):
    observed_values = data
    observed_masks = ~tf.math.is_nan(data)
    gt_masks = observed_masks
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)
    gt_masks = tf.Variable(gt_masks,trainable=False)
    for channel in range(gt_masks.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        idx = [i for i in range(s_nan[0],s_nan[-1] + 1)]
        tensor=tf.tensor_scatter_nd_update(gt_masks[:,channel],tf.expand_dims(idx,axis = 1),[0]*(len(idx)))
        gt_masks[:,channel].assign(tensor)
    
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(observed_values)), dtype=tf.float32)
    observed_values = tf.math.multiply_no_nan(observed_values, value_not_nan)
    
    return observed_values, observed_masks, tf.convert_to_tensor(gt_masks)


def mask_missing_train_bm(data, k_segments=5):
    observed_values = data
    observed_masks = ~tf.math.is_nan(data)
    gt_masks = observed_masks
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)
    s_nan = random.choice(list_of_segments_index)
    idx = [i for i in range(s_nan[0],s_nan[-1] + 1)]
    gt_masks = tf.Variable(gt_masks,trainable=False)
    for channel in range(gt_masks.shape[1]):
        tensor=tf.tensor_scatter_nd_update(gt_masks[:,channel],tf.expand_dims(idx,axis = 1),[0]*(len(idx)))
        gt_masks[:,channel].assign(tensor)
        
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(observed_values)), dtype=tf.float32)
    observed_values = tf.math.multiply_no_nan(observed_values, value_not_nan)
   
    return observed_values, observed_masks, tf.convert_to_tensor(gt_masks)




def get_dataloader_train_impute(series,
                                batch_size=8,
                                missing_ratio_or_k=0.2,
                                len_dataset=248,
                                masking='rm',
                               path_save='',
                               ms=None):
   
 
    count = 0
    drop_value = series.shape[0]%batch_size
    series = tf.convert_to_tensor(series[drop_value:])
    
    for sample in series:
        if masking == 'rm':
            observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)
        elif masking == 'nrm':
            observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)
        elif masking == 'bm':
            observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)

        if count == 0:
            all_observed_values = observed_values
            all_observed_masks = observed_masks
            all_gt_masks = gt_masks
            all_timepoints = tf.range(start=0, limit=len_dataset,delta=1)
            count += 1
        else:
            all_observed_values = tf.concat([all_observed_values, observed_values],axis=0)
            all_observed_masks = tf.concat([all_observed_masks, observed_masks],axis=0)
            all_gt_masks = tf.concat([all_gt_masks , gt_masks],axis=0)
            all_timepoints = tf.concat([all_timepoints,  tf.range(start=0, limit=len_dataset,delta=1)],axis=0)
            
            
    
    return (tf.reshape(tf.convert_to_tensor(all_observed_values),(-1,sample.shape[0],sample.shape[1])),
            tf.reshape(tf.convert_to_tensor(all_observed_masks),(-1,sample.shape[0],sample.shape[1])), 
            tf.reshape(tf.convert_to_tensor(all_gt_masks),(-1,sample.shape[0],sample.shape[1])),
            tf.reshape(tf.convert_to_tensor(all_timepoints),(-1,sample.shape[0])))



class CSDIImputer:
    def __init__(self,gpu):
        np.random.seed(0)
        random.seed(0)
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(gpus[0],'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
      
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              #series,
              masking ='rm',
              missing_ratio_or_k = 0.2,
              trainset_path = './datasets/train_ptbxl_248.npy',
              valset_path = './datasets/val_ptbxl_248.npy',
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
              target_strategy = 'random',
             ):
        
        '''
        CSDI training function. 
       
       
        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masking: 'rm': random missing, 'nrm': non-random missing, 'bm': black-out missing.
        -missing_ratio_or_k: missing ratio 0 to 1 for 'rm' masking and k segments for 'nrm' and 'bm'.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.
        
        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions. 
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler. 
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking. 
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity. 
        '''
       
        config = {}
        
        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
      
        config['train']['path_save'] = path_save
        
       
        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule
        
        config['model'] = {} 
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking
        
        print(json.dumps(config, indent=4))

        config_filename = path_save + "config_csdi_training"
        print('configuration file name:', config_filename)
        with open(config_filename + ".json", "w") as f:
            json.dump(config, f, indent=4)
            
        

        training_data = np.load(trainset_path) 
        training_data = np.transpose(training_data,(0,2,1))
        
        validation_data = np.load(valset_path) 
        validation_data = np.transpose(validation_data,(0,2,1))
     
        
        testing_data = np.load(testset_path) 
        testing_data = np.transpose(testing_data,(0,2,1))
       


        train_loader = get_dataloader_train_impute(series=training_data,
                                                   len_dataset=training_data.shape[1],
                                                    batch_size=config["train"]["batch_size"],
                                                    missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
                                                    masking=config['model']['masking'],
                                                    path_save=config['train']['path_save'])
                            
                            
        val_loader = get_dataloader_train_impute(series=validation_data,
                                                  len_dataset=validation_data.shape[1],
                                                    batch_size=config["train"]["batch_size"],
                                                   missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
                                                    masking=config['model']['masking'],
                                                    path_save=config['train']['path_save'])
        
        
        
        test_loader = get_dataloader_train_impute(series=testing_data,
                                                   len_dataset=testing_data.shape[1],
                                                    batch_size=config["train"]["batch_size"],
                                                    missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
                                                    masking=config['model']['masking'],
                                                    path_save=config['train']['path_save'])
        print('finish data processing')
    
    
        model = CSDI_base(config, target_dim=training_data.shape[2])
        
        train(model=model,
              config=config["train"],
              train_loader=train_loader,
             val_loader=val_loader,
              path_save=config['train']['path_save'],
              batch_size=config["train"]["batch_size"])

        evaluate(model=model,
                 test_loader=test_loader,
                 nsample=samples_generate,
                 scaler=1,
                 path_save=config['train']['path_save'],
                batch_size=config["train"]["batch_size"])
        
        
  
  
