
import tensorflow as tf
from einops import rearrange
from imputers.S4Model import S4, LinearActivation
from utils.util import calc_diffusion_step_embedding





class DownPool(tf.keras.Model):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def call(self, x):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = tf.transpose(self.linear(tf.transpose(x,[0,2,1])),[0,2,1])
        return x



class UpPool(tf.keras.Model):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def call(self, x):
        x = tf.transpose(self.linear(tf.transpose(x,[0,2,1])),[0,2,1])
        if(self.causal):
            x = tf.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        return x



class FFBlock(tf.keras.Model):

    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        self.input_linear = LinearActivation(
            d_model, 
            d_model * expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        self.dropout = tf.keras.layers.SpatialDropout2D(dropout) if dropout > 0.0 else tf.identity
        self.output_linear = LinearActivation(
            d_model * expand,
            d_model, 
            transposed=True,
            activation=None,
            activate=False,
        )
        

    def call(self, x):
        x = tf.transpose(x,[0,2,1])
        for module in self.input_linear:
            x = module(x)
        x = tf.transpose(x,[0,2,1])
        x = self.dropout(x)
        
        x = tf.transpose(x,[0,2,1])
        x = self.output_linear(x)
        x = tf.transpose(x,[0,2,1])
  
        return x, None



class ResidualBlock(tf.keras.Model):

    def __init__(
        self, 
        d_model, 
        layer,
        dropout,
        diffusion_step_embed_dim_out,
        in_channels,
        stride
    ):
        
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)
        self.dropout = tf.keras.layers.SpatialDropout2D(dropout) if dropout > 0.0 else tf.identity

        self.fc_t = tf.keras.layers.Dense(d_model)
        
        self.cond_conv = tf.keras.layers.Conv1D(filters=d_model, kernel_size=stride, strides=stride, padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')
        
        
        
    def call(self, input_data):
        """
        Input x is shape (B, d_input, L)
        """
        x, cond, diffusion_step_embed = input_data
        
        # add in diffusion step embedding
        part_t = tf.expand_dims(self.fc_t(diffusion_step_embed),2)
        z = x + part_t
        
        # Prenorm
        z = self.norm(z)
        
        z,_ = self.layer(z) 
        
        
        cond = self.cond_conv(cond)
       
        z = z + cond
            
        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x


class SSSDSAImputer(tf.keras.Model):
    def __init__(
        self,
        d_model=64, 
        n_layers=6, 
        pool=[2, 2], 
        expand=2, 
        ff=2, 
        glu=True,
        unet=True,
        dropout=0.0,
        in_channels=1,
        out_channels=1,
        diffusion_step_embed_dim_in=128, 
        diffusion_step_embed_dim_mid=512,
        diffusion_step_embed_dim_out=512,
        bidirectional=True,
        s4_lmax=1,
        s4_d_state=64,
        s4_dropout=0.0,
        s4_bidirectional=True,
    ):
        
        """
        SaShiMi model backbone. 

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level. 
                We use 8 layers for our experiments, although we found that increasing layers even further generally 
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels. 
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models 
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling. 
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet

        def s4_block(dim, stride):
          
            layer = S4(
                d_model=dim, 
                l_max=s4_lmax,
                d_state=s4_d_state,
                bidirectional=s4_bidirectional,
                postact='glu' if glu else None,
                dropout=dropout,
                transposed=True,
                trainable={
                    'dt': True,
                    'A': True,
                    'P': True,
                    'B': True,
                }, # train all internal S4 parameters
                    
            )
            
                
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                stride=stride     
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels = in_channels,
                stride=stride
            )

        # Down blocks
       
        d_layers = []
        for i, p in enumerate(pool):
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    if i == 0:
                        d_layers.append(s4_block(H, 1))
                        if ff > 0: d_layers.append(ff_block(H, 1))
                    elif i == 1:
                        d_layers.append(s4_block(H, p))
                        if ff > 0: d_layers.append(ff_block(H, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        
        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, pool[1]*2))
            if ff > 0: c_layers.append(ff_block(H, pool[1]*2))
        
        # Up blocks
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal= not bidirectional))

            for _ in range(n_layers):
                if i == 0:
                    block.append(s4_block(H, pool[0]))
                    if ff > 0: block.append(ff_block(H, pool[0]))
                        
                elif i == 1:
                    block.append(s4_block(H, 1))
                    if ff > 0: block.append(ff_block(H, 1))

           # u_layers.append(nn.ModuleList(block))
            u_layers.append(block)
        
        self.d_layers = d_layers
        self.c_layers = c_layers
        self.u_layers = u_layers
        
        self.norm = tf.keras.layers.LayerNormalization(axis=-2)
        
        self.init_conv=[tf.keras.layers.Conv1D(filters=d_model, kernel_size=1,padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first'),tf.keras.layers.ReLU()]
        self.final_conv=[tf.keras.layers.Conv1D(filters=d_model, kernel_size=1,padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first'),tf.keras.layers.ReLU(),tf.keras.layers.Conv1D(filters=out_channels, kernel_size=1,padding = 'SAME',kernel_initializer='he_normal', data_format='channels_first')]
        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)
 
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        assert H == d_model

    def call(self, input_data):

        noise, conditional, mask, diffusion_steps,y_true, loss_mask = input_data
        
        
        conditional = conditional * mask     
        conditional = tf.concat([conditional, tf.cast(mask,dtype=tf.float32)], axis=1)
          
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        
        diffusion_step_embed = tf.keras.activations.swish((self.fc_t1(diffusion_step_embed)))
        diffusion_step_embed = tf.keras.activations.swish((self.fc_t2(diffusion_step_embed)))

        x = noise    
        for module in self.init_conv:
            x = module(x)
 
        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer((x,conditional,diffusion_step_embed))
            else:
                x = layer(x)
            outputs.append(x)
            
        # Center block
        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer((x,conditional,diffusion_step_embed))
            else:
                x = layer(x)
        
        
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x,conditional,diffusion_step_embed))
                    else:
                        x = layer(x)
                    x = x + outputs.pop() # skip connection
            else:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x,conditional,diffusion_step_embed))
                    else:
                        x = layer(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = self.norm(x)
        
        for module in self.final_conv:
            x = module(x)
        
        
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(y_true[loss_mask],x[loss_mask])
        self.add_loss(loss)
        
        return x 

 
