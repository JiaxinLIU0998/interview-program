import math
from utils.util import calc_diffusion_step_embedding
from imputers.S5 import SequenceLayer
import os
import tensorflow as tf



class Residual_block(tf.keras.Model):
    """Implementation of each residual block for the residual group """
    
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
               ):
        """Initializer.
        Args:
            hyperparameter defined by 'wavenet config'
        """
        
        super(Residual_block, self).__init__()
        
        self.res_channels = res_channels
        self.fc_t = tf.keras.layers.Dense(self.res_channels)
      
                            
        self.mega1 = SequenceLayer(2*self.res_channels)
      
        

        self.conv_layer = tf.keras.layers.Conv1D(filters=2 * self.res_channels, kernel_size=3, padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')
        
        self.mega2 = SequenceLayer(2*self.res_channels)
     

        self.cond_conv = tf.keras.layers.Conv1D(filters=2*self.res_channels, kernel_size=1, padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')
 
        self.res_conv1 = tf.keras.layers.Conv1D(filters=res_channels, kernel_size=1, padding = 'SAME', kernel_initializer='he_normal', data_format='channels_first')
        self.res_conv2 = tf.keras.layers.Conv1D(filters=skip_channels, kernel_size=1, padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')

      
    def call(self, input_data,training):
        """Pass to Residual_block.
        Args:
            inputs: tuple (x, cond, diffusion_step_embed)
                    x: noised input [B,C,L]
                    cond: conditioning information [B,2*K,L]
                    diffusion_step_embed: [B, 512]
        Returns:
            outputs: two tf.Tensor, [B, skip_channels, L], [B, skip_channels, L].
        """
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = h.shape
        assert C == self.res_channels                      

        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t,[B, self.res_channels, 1])    
        h = h + part_t
        
        h = self.conv_layer(h)
        h = self.mega1(tf.transpose(h,(0,2,1)),training=training)
        h = tf.transpose(h,(0,2,1))
                       

        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond

        h = self.mega2(tf.transpose(h,(0,2,1)),training=training)
        h = tf.transpose(h,(0,2,1))
        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv1(out)
        assert x.shape == res.shape
        skip = self.res_conv2(out)

        return (x + res) * tf.math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(tf.keras.Model):
    """Implementation of residual group component for SSSD(S4) model """
    
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 ):
        """Initializer.
        Args:
            hyperparameter defined by 'wavenet config'
        """
        
        super(Residual_group, self).__init__()
        
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)

        self.residual_blocks = []

        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       ))
           

    def call(self, input_data,training):
        """Pass to Residual_group.
        Args:
            inputs: tuple (h, conditional, diffusion_steps)
                    h: noised input [B,C,L]
                    conditional: conditioning information [B,2*K,L]
                    diffusion_steps: [B, 1]
        Returns:
            outputs: tf.Tensor, [B, skip_channels, L], output tensor.
        """
       
        h, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)

        diffusion_step_embed = tf.keras.activations.swish((self.fc_t1(diffusion_step_embed)))
        diffusion_step_embed = tf.keras.activations.swish((self.fc_t2(diffusion_step_embed)))

        skip = 0.0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed),training=training)  
            skip += skip_n  

        return skip * tf.math.sqrt(1.0 / self.num_res_layers)


class SSSDS5Imputer(tf.keras.Model):
    """Implementation of SSSD(S4) model """
    
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 ):
        """Initializer.
        Args:
            hyperparameter defined by 'wavenet config'
        """
        super(SSSDS5Imputer, self).__init__()
        
       
        # convert the dimension of input from (B,in_channels,L) to (B,res_channels,L)
        self.init_conv = tf.keras.layers.Conv1D(filters=res_channels, kernel_size=1, padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             )
        # convert the dimension from (B,skip_channels,L) to (B,out_channels,L)
        self.final_conv = tf.keras.Sequential()
        self.final_conv.add(tf.keras.layers.Conv1D(filters=skip_channels, kernel_size=1, padding = 'SAME', kernel_initializer='he_normal', data_format='channels_first'))
        
        self.final_conv.add(tf.keras.layers.ReLU()) 
        self.final_conv.add(tf.keras.layers.Conv1D(filters=out_channels, kernel_size=1, padding = 'SAME', kernel_initializer='zeros',data_format='channels_first'))
        
        
    

    def call(self, input_data,training=True):
        """Pass to SSSDS4Imputer.
        Args:
            inputs: tuple (x, conditional, mask, diffusion_steps)
                    x: noised input [B,K,L]
                    conditional: conditioning information [B,K,L]
                    mask: [B,K,L]
                    diffusion_steps: [B, 1]
        Returns:
            outputs: tf.Tensor, [B, K, L], output tensor.
        """

        h, conditional, mask, diffusion_steps,y_true, loss_mask = input_data

        conditional = conditional * mask
        conditional = tf.concat([conditional, tf.cast(mask,dtype=tf.float32)], axis=1)

        x = tf.nn.relu(self.init_conv(h))
        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == self.res_channels
        assert x.shape[2] == h.shape[2]

        x = self.residual_layer((x, conditional, diffusion_steps),training=training)
        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == self.skip_channels
        assert x.shape[2] == h.shape[2]
        
        y = self.final_conv(x)

        assert y.shape[0] == h.shape[0]
        assert y.shape[1] == self.out_channels
        assert y.shape[2] == h.shape[2]

        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(y_true[loss_mask],y[loss_mask])
        self.add_loss(loss)
        
        return y

                
    def predict(self, input_data,batch_size,training=False):
        """Pass to SSSDS4Imputer.
        Args:
            inputs: tuple (x, conditional, mask, diffusion_steps)
                    x: noised input [B,K,L]
                    conditional: conditioning information [B,K,L]
                    mask: [B,K,L]
                    diffusion_steps: [B, 1]
        Returns:
            outputs: tf.Tensor, [B, K, L], output tensor.
        """

        h, conditional, mask, diffusion_steps,y_true, loss_mask = input_data

        conditional = conditional * mask
        conditional = tf.concat([conditional, tf.cast(mask,dtype=tf.float32)], axis=1)

        x = tf.nn.relu(self.init_conv(h))
        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == self.res_channels
        assert x.shape[2] == h.shape[2]

        x = self.residual_layer((x, conditional, diffusion_steps),training=training)
        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == self.skip_channels
        assert x.shape[2] == h.shape[2]
        
        y = self.final_conv(x)

        assert y.shape[0] == h.shape[0]
        assert y.shape[1] == self.out_channels
        assert y.shape[2] == h.shape[2]

        #loss_fn = tf.keras.losses.MeanSquaredError()
        #loss = loss_fn(y_true[loss_mask],y[loss_mask])
        #self.add_loss(loss)
        
        return y
