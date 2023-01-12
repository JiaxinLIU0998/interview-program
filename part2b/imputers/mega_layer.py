# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union, Optional,Dict,Callable
import math
import tensorflow as tf
import uuid
import warnings


def gelu_accurate(x):
    return tf.cast(tf.keras.activations.gelu(tf.cast(x,tf.float32),approximate=True),x.dtype)


def gelu(x):
    return tf.cast(tf.keras.activations.gelu(tf.cast(x,tf.float32),approximate=False),x.dtype)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return tf.keras.layers.ReLU()
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == 'silu':
        return tf.keras.layers.Activation(tf.nn.silu)
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def relu2(x, onnx_trace: bool = False):
    if onnx_trace:
        return tf.math.square(tf.nn.relu(tf.cast(x,tf.float32)))
    else:
        return tf.math.square(tf.nn.relu(x))



def laplace(x, mu=0.707107, sigma=0.282095, onnx_trace: bool = False):
    if onnx_trace:
        x = tf.cast(x,tf.float32)
    x = tf.math.divide((x - mu),sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + tf.math.erf(x))


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return tf.nn.softmax(tf.cast(x,tf.float32), axis=dim)
    else:
        return tf.cast(tf.nn.softmax(x, axis=dim),dtype=tf.float32)

    


class FairseqIncrementalState(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[tf.Tensor]]]:
        """Helper for getting incremental state for an tf.keras.Model."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]],
        key: str,
        value: Dict[str, Optional[tf.Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]]:
        """Helper for setting incremental state for an tf.keras.Model."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls


@with_incremental_state
class MultiHeadEMA(tf.keras.Model):
    """Exponential Moving Average Layer.
    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        
        self.delta = tf.Variable(tf.random.normal([kernel_dim, ndim, 1],mean=0.0,stddev=0.2),trainable=True)
        self.alpha = tf.Variable(tf.random.normal([kernel_dim, ndim, 1],mean=0.0,stddev=0.2),trainable=True)
        
        val = tf.ones([self.ndim, 1])
        if self.ndim > 1:
            val = tf.expand_dims(tf.convert_to_tensor([1 if i%1==0 else -1 for i in range(self.ndim)]),1)
        
        self.beta = tf.Variable(tf.random.normal([kernel_dim, ndim, 1],mean=0.0,stddev=0.02)+tf.cast(val,tf.float32),trainable=True)
        
        self.gamma = tf.Variable(tf.random.normal([kernel_dim, ndim],mean=0.0,stddev=1.0),trainable=True)
        self.omega = tf.Variable(tf.random.normal([embed_dim],mean=0.0,stddev=1.0),trainable=True)
        
        self._kernel = None
        self._coeffs = None
        
        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = tf.math.sigmoid(self.delta)
        alpha = tf.math.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        #vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        vander = tf.reshape(tf.cast(tf.range(start=0,delta=1,limit = length),p.dtype),(1, 1, length)) * tf.math.log(q)
        kernel = (p * self.beta) * tf.math.exp(vander)
        #kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        #return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)
        return tf.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.shape[-1] < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def step(self, x, length, hx=None):
        if length == 1:
            return self.one_step(x, hx=hx)

        # D x N x 1
        p, q = self.coeffs()
        # D x N x L+1
        vander = tf.reshape(tf.cast(tf.range(start=0,delta=1,limit = length + 1),p.dtype),(1, 1, length + 1)) * tf.math.log(q)
        #vander = torch.arange(length + 1).to(p).view(1, 1, length + 1) * torch.log(q)
        vander = tf.math.exp(vander)
        if hx is not None:
            # D x N x L * D x N x 1 -> D x N x L
            k = vander[:, :, 1:] * tf.expand_dims((self.gamma * self.scale),-1)
            #k = vander[:, :, 1:] * (self.gamma * self.scale).unsqueeze(-1)
            #ox = torch.einsum('bdn,dnl->bdl', hx, k)
            ox = tf.einsum('bdn,dnl->bdl', hx, k)
            # D x N * B x D x N -> B x D x N
            hh = vander[:, :, -1] * hx
        else:
            ox = None
            hh = None

        # D x N x L
        vander = vander[:, :, :-1]
        kernel = (p * self.beta) * vander
        #k = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)
        k = tf.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

        k_f = tf.signal.rfft(tf.cast(k,tf.float32), fft_length=2 * length)
        x_f = tf.signal.rfft(tf.cast(x,tf.float32), fft_length=2 * length)
        # B x D x L
        out = tf.signal.irfft(x_f * k_f, fft_length=2 * length)[..., 0:length]
        out = tf.cast(out,x.dtype)
       # out = out.type_as(x)
        if ox is not None:
            out = out + ox

        #h = torch.einsum('bdl,dnl->bdn', x, torch.flip(kernel, dims=[2]))
        h = tf.einsum('bdl,dnl->bdn', x, tf.reverse(kernel, axis=2))
        if hh is not None:
            h = h + hh
        # L x B x D, B x D x N
        return tf.transpose(out,(2, 0, 1)), h
        #return out.permute(2, 0, 1), h

    def one_step(self, x, hx=None):
        p, q = self.coeffs()
        # (D x N) x (B x D x 1) -> B x D x N
        h = tf.squeeze(p * self.beta,-1)* x
        #h = (p * self.beta).squeeze(-1) * x
        if hx is not None:
            h = h + tf.squeeze(q * self.beta,-1) * hx
            #h = h + q.squeeze(-1) * hx
        # B x D
        out = tf.einsum('bdn,dn->bd', h, self.gamma * self.scale)
        # 1 x B x D, B x D x N
        #return out.unsqueeze(0), h
        
        return tf.expand_dims(out,1), h

    def call(
        self,
        x,
        padding_mask: Optional[tf.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]] = None,
        training = True,
    ) -> tf.Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        self.training = training
        seq_len, bsz, embed_dim = x.shape[0],x.shape[1],x.shape[2]
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = tf.transpose(x,(1, 2, 0))
        #x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - tf.cast(tf.expand_dims(padding_mask,1),x.dtype))
            #x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        assert not self.bidirectional or incremental_state is None, 'Bidirectional EMA does not support incremental state'
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_state' in saved_state:
                h = saved_state['prev_state']
            else:
                h = None
            out, h = self.step(x, seq_len, hx=h)
            saved_state['prev_state'] = h
            self._set_input_buffer(incremental_state, saved_state)
            # B x D -> 1 x B x D
            
            out = tf.nn.silu(out + residual)
        else:
            # D x L
            k = self.kernel(seq_len)
            fft_len = seq_len
            s = 0
            kernel_size = k.shape[1]
            if self.bidirectional:
                k1, k2 = tf.split(k, [self.embed_dim, self.embed_dim], 0)
                # D x 2*L-1
                
                #k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
                #x = F.pad(x, (kernel_size - 1, 0))
                k = tf.pad(k1, [[0,0],[kernel_size - 1, 0]]) + tf.pad(tf.reverse(k2,axis=[-1]), [[0,0],[0, kernel_size - 1]])
                x = tf.pad(x, [[0,0],[0,0],[kernel_size - 1, 0]])
                
                #k = tf.pad(k1, (kernel_size - 1, 0)) + tf.pad(tf.reverse(k2,-1), (0, kernel_size - 1))
                #x = tf.pad(x, (kernel_size - 1, 0))
                
                fft_len = fft_len + kernel_size - 1
                s = 2 * kernel_size - 2
            
            k_f = tf.signal.rfft(tf.cast(k,tf.float32), fft_length=tf.convert_to_tensor([2 * fft_len],dtype=tf.int32))
            x_f = tf.signal.rfft(tf.cast(x,tf.float32), fft_length=tf.convert_to_tensor([2 * fft_len],dtype=tf.int32))
            # B x D x L
            out = tf.signal.irfft(x_f * k_f, fft_length=tf.convert_to_tensor([2 * fft_len],dtype=tf.int32))[..., s:s + seq_len]
            out = tf.cast(out,x.dtype)
            out = tf.nn.silu(tf.transpose(out,(2, 0, 1)) + residual)

        return out

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]]) -> Dict[str, Optional[tf.Tensor]]:
        result = self.get_incremental_state(incremental_state, "ema_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[tf.Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[tf.Tensor]]], buffer: Dict[str, Optional[tf.Tensor]]):
        return self.set_incremental_state(incremental_state, "ema_state", buffer)

   # @torch.jit.export
   # def reorder_incremental_state(
   #         self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
   # ):
   #     """Reorder buffered internal state (for incremental generation)."""
   #     input_buffer = self._get_input_buffer(incremental_state)
   #     if input_buffer is not None:
   #         for k in input_buffer.keys():
   #             input_buffer_k = input_buffer[k]
   #             if input_buffer_k is not None:
   #                 input_buffer[k] = input_buffer_k.index_select(0, new_order)
   #         incremental_state = self._set_input_buffer(incremental_state, input_buffer)
   #     return incremental_state

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}'.format(self.embed_dim, self.ndim, self.bidirectional, self.truncation)


class SimpleRelativePositionalBias(tf.keras.Model):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = tf.Variable(tf.random.normal([2 * max_positions - 1],mean=0.0,stddev=0.02),trainable=True)
  
    def call(self, seq_len):
        # seq_len * 2 -1
        b = self.rel_pos_bias[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = tf.pad(b, [[0, seq_len]])
        # (seq_len * 3 - 1) * seq_len
        #t = torch.tile(t, (seq_len,))
        t = tf.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = tf.reshape(t,(seq_len, 3 * seq_len - 2))
        #t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.shape[1] - r
        t = t[:, start:end]
        return t

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)


class RotaryRelativePositionalBias(tf.keras.Model):
    def __init__(self, embed_dim, max_positions):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, embed_dim)
        
        self.beta = tf.Variable(tf.random.normal([1, embed_dim],mean=0.0,stddev=0.02),trainable=True)
        self.alpha = tf.Variable(tf.random.normal([1, embed_dim],mean=0.0,stddev=0.02),trainable=True)
        
     
        self._float_tensor = tf.Variable(tf.random.normal([1],mean=0.0,stddev=0.02),name = "_float_tensor",trainable=False)
  

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = tf.math.exp(tf.range(limit = half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(tf.range(limit = max_positions, dtype=tf.float32),1) * tf.expand_dims(emb,0)
        #emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return tf.math.sin(emb), tf.math.cos(emb)

    def rotary(self, x):
        n, d = x.shape[0],x.shape[1]
        
        x1, x2 = tf.split(x, 2, -1)
        if self.sine is None or n > self.sine.shape[0]:
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = tf.cast(self.sine,self._float_tensor.dtype)
        self.cosine = tf.cast(self.cosine,self._float_tensor.dtype)
        sin = self.sine[:n]
        cos = self.cosine[:n]
        return tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], 1)

    def call(self, seq_len):
        a = self.rotary(tf.broadcast_to(self.alpha,(seq_len, self.embed_dim)))
        b = self.rotary(tf.broadcast_to(self.beta,(seq_len, self.embed_dim)))
        t = tf.einsum('mk,nk->mn', a, b)
        return t

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}'.format(self.embed_dim, self.max_positions)

    

class RMSNorm(tf.keras.Model):
    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = tf.Variable(tf.ones([self.num_features]),trainable=True)
            #self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.weight = tf.Variable(name = 'weight',trainable=True)
 

    def call(self, x):
        
        mean_square = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        if self.weight is not None:
            x = x * self.weight

        x = x * tf.math.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return '{num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)





class ScaleNorm(tf.keras.Model):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = tf.Variable(tf.ones([1]),trainable=True)
            #self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.scalar = tf.Variable(name = 'scalar',trainable=True)
            
 

    def call(self, x):
        mean_square = tf.math.reduce_mean(tf.math.square(x), axis=self.dim, keepdims=True)
        if self.scalar is not None:
            x = self.scalar * x

        x = x * tf.math.rsqrt(mean_square + self.eps)
        return x


    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}, affine={affine}'.format(**self.__dict__)





class SequenceNorm(tf.keras.Model):
    def __init__(self, norm_type, embedding_dim, eps=1e-5, affine=True, export=False):
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = tf.keras.layers.LayerNormalization(axis=-1,epsilon=eps)
            #self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=affine, export=export)
        elif norm_type == 'scalenorm':
            self.norm = ScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = tf.keras.layers.BatchNormalization(axis=-1,epsilon=eps)
            #self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1,epsilon=eps)
            #self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def normalize(self, x):
   
        return self.norm(x)

    def call(self, x):
        return self.normalize(x)



class NormalizedFeedForwardNetwork(tf.keras.Model):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        export=False,
    ):
        super().__init__()

        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = activation
        self.activation = get_activation_fn(activation)

        
       
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)
        self.fc1 = tf.keras.layers.Dense(ffn_hidden_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),bias_initializer='zeros')
        self.fc2 = tf.keras.layers.Dense(embed_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),bias_initializer='zeros')
       
        

    def call(self, x):
        residual = x

        if self.prenorm:
            x = self.norm(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        if not self.prenorm:
            x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return 'edim={}, hdim={}, act={}, prenorm={}'.format(self.embedding_dim, self.hidden_dim, self.act_fn, self.prenorm)

    

@with_incremental_state
class MovingAverageGatedAttention(tf.keras.Model):
    """Exponential Moving Average Gated Attention.
    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        attention_activation='softmax',
        bidirectional=False,
        chunk_size=-1,
        truncation=None,
        norm_type='layernorm',
        prenorm=True,
        norm_affine=True,
        feature_dropout=False,
        rel_pos_bias='simple',
        max_positions=1024,
        export=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

      
        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.norm = SequenceNorm(norm_type, embed_dim, affine=norm_affine, export=export)

        self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        
        self.v_proj = tf.keras.layers.Dense(hdim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),bias_initializer='zeros')
        self.mx_proj = tf.keras.layers.Dense(zdim + hdim + 2 * embed_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),bias_initializer='zeros')
        self.h_proj = tf.keras.layers.Dense(embed_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),bias_initializer='zeros')
        
        

        self.gamma = tf.Variable(tf.random.normal([2, zdim],mean=0.0,stddev=0.02),trainable=True)
        self.beta = tf.Variable(tf.zeros([2, zdim]),trainable=True)
   

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        #self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

 

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.shape[2]
        if padding_mask is not None:
            # B x K x C
            inverse_mask = 1.0 -  tf.cast(padding_mask,q.dtype)
            # B x K x 1
            lengths = tf.reduce_sum(inverse_mask,axis=-1,keepdims=True)
            # B x K x 1 x 1
            
            lengths = tf.expand_dims(tf.clip_by_value(lengths, clip_value_min=1.0, clip_value_max=tf.math.reduce_max(lengths)),-1)
        else:
            lengths = slen
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            
            lengths = tf.reduce_sum(attn_mask,axis=-1,keepdims=True)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.shape[2]:
            assert q.shape[2] == 1
            # 1 x C
            bias = bias[-1:]

        # B x K x C x C
        
        qk = tf.matmul(q, tf.transpose(k,(0,1,3, 2))) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = tf.cast(relu2(qk),qk.dtype)
        elif self.attention_activation == 'laplace':
            attn_weights = tf.cast(laplace(qk),qk.dtype)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * tf.expand_dims(inverse_mask,2)

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.shape[2]
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.shape[2]:
            assert q.shape[2] == 1
            # 1 x C
            bias = bias[-1:]

        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = tf.matmul(q, tf.transpose(k,(0,1,3,2))) + bias

        if attn_mask is not None:
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = tf.math.reduce_all(padding_mask,axis=-1,keepdims=True,)
            padding_mask = tf.logical_and(padding_mask, tf.logical_not(padding_mask_all))
            qk = tf.where(tf.cast(tf.expand_dims(padding_mask,2),tf.bool),float('-inf'),qk)
            #qk = qk.masked_fill(tf.cast(tf.expand_dims(padding_mask,2),tf.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = tf.cast(softmax(qk, dim=-1, onnx_trace=self.onnx_trace),qk.dtype)
        return attn_weights

    def call(
        self,
        x,
        padding_mask: Optional[tf.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]] = None,
        need_weights: bool = False,
        attn_mask: Optional[tf.Tensor] = None,
        before_attn_fn: bool = False,
        training = True,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """
        self.training = training
        seq_len, bsz, embed_dim = x.shape[0],x.shape[1],x.shape[2]
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        residual = x
        if self.prenorm:
            x = self.norm(x)

        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        mx = self.move(x, padding_mask, incremental_state,self.training)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = tf.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], axis=-1)
        # L x B x D
        u = tf.math.sigmoid(u)
        # L x B x (E+S)
        z, r = tf.split(tf.nn.silu(zr), [self.zdim, self.hdim], axis=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = tf.expand_dims(z,2) * self.gamma + self.beta
        # L x B x 2 x S -> L x B x S
        q, k = tf.unstack(z, axis=2)

        # L x B x D -> B x L x D
        q = tf.transpose(q,(1,0,2))
        k = tf.transpose(k,(1,0,2))
        v = tf.transpose(v,(1,0,2))
       

        if saved_state is not None:
            # assert self.chunk_size < 0 or q.size(1) <= self.chunk_size
            # saved states are stored with shape (bsz, seq_len, dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                assert k is not None
                k = tf.concat([prev_key, k], axis=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                assert v is not None
                v = tf.concat([prev_value, v], axis=1)
            prev_padding_mask: Optional[tf.Tensor] = None
            if "prev_padding_mask" in saved_state:
                prev_padding_mask = saved_state["prev_padding_mask"]
            padding_mask = MovingAverageGatedAttention._append_prev_padding_mask(
                padding_mask=padding_mask,
                prev_padding_mask=prev_padding_mask,
                batch_size=bsz,
                seq_len=k.shape[1],
            )

            if self.chunk_size < 0:
                saved_state["prev_key"] = k
                saved_state["prev_value"] = v
                saved_state["prev_key_padding_mask"] = padding_mask
            else:
                curr_len = k.shape[1] % self.chunk_size
                if curr_len == 0:
                    if "prev_key" in saved_state:
                        del saved_state["prev_key"]
                        del saved_state["prev_value"]
                        del saved_state["prev_key_padding_mask"]
                else:
                    saved_state["prev_key"] = k
                    saved_state["prev_value"] = v
                    saved_state["prev_key_padding_mask"] = padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        ctx_len = k.shape[1]
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            q = tf.expand_dims(q,1)
            k = tf.expand_dims(k,1)
            v = tf.expand_dims(v,1)
            
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = tf.expand_dims(padding_mask,1)
            
        else:
            if seq_len < self.chunk_size:
                 q = tf.expand_dims(q,1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                q = tf.reshape(q,(bsz, nc, self.chunk_size, self.zdim))

            if ctx_len < self.chunk_size:
                k = tf.expand_dims(k,1)
                v = tf.expand_dims(v,1)
                if padding_mask is not None:
                    padding_mask = tf.expand_dims(padding_mask,1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
               
                k = tf.reshape(k,(bsz, nc, self.chunk_size, self.zdim))
                v = tf.reshape(v,(bsz, nc, self.chunk_size, self.hdim))
                if padding_mask is not None:
                    # B x L -> B x K x C
                    padding_mask = tf.reshape(padding_mask,(bsz, nc, self.chunk_size))

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and len(padding_mask.get_shape().as_list()) == 0:
            padding_mask = None

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        kernel = attn_weights
        # B x K x C x E -> B x L x E -> L x B x E
        
        h = tf.transpose(tf.reshape(tf.matmul(kernel, v),(bsz, seq_len, self.hdim)),(1,0, 2))
        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        # L x B x D

        out = residual+tf.multiply( u, h - residual)
        #out = torch.addcmul(residual, u, h - residual)

        if not self.prenorm:
            out = self.norm(out)

        if need_weights:
            return out, attn_weights
        else:
            return out, None

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[tf.Tensor]]]]) -> Dict[str, Optional[tf.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[tf.Tensor]] = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[tf.Tensor]]], buffer: Dict[str, Optional[tf.Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

   # @torch.jit.export
   # def reorder_incremental_state(
   #         self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
   # ):
   #     """Reorder buffered internal state (for incremental generation)."""
   #     input_buffer = self._get_input_buffer(incremental_state)
   #     if input_buffer is not None:
   #         for k in input_buffer.keys():
   #             input_buffer_k = input_buffer[k]
   #             if input_buffer_k is not None:
   #                 input_buffer[k] = input_buffer_k.index_select(0, new_order)
   #         incremental_state = self._set_input_buffer(incremental_state, input_buffer)
   #     return incremental_state

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[tf.Tensor],
        prev_padding_mask: Optional[tf.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[tf.Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = tf.concat([prev_padding_mask, padding_mask], axis=1)
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = tf.zeros((batch_size, seq_len - prev_padding_mask.shape[1]))
            new_padding_mask = tf.concat([prev_padding_mask, tf.cast(filler,tf.bool())], axis=1)
        elif padding_mask is not None:
            filler = tf.zeros((batch_size, seq_len - padding_mask.shape[1]))
            new_padding_mask = tf.concat([tf.cast(filler,tf.bool()), padding_mask], axis=1)
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, ndim={}, chunk={}, attn_act={}, prenorm={}'.format(self.embed_dim, self.zdim,
                                                                                  self.hdim, self.ndim, self.chunk_size,
                                                                                  self.attention_activation, self.prenorm)



class MegaEncoderLayer(tf.keras.Model):
    """Encoder layer block.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, encoder_embed_dim):
        super().__init__()
        self.embed_dim = encoder_embed_dim
        self.mega_layer = self.build_mega_layer(self.embed_dim)
        self.nffn = self.build_nffn_layer(self.embed_dim)
      
    def build_mega_layer(self, embed_dim):
        return MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=30,
            hdim=120,
            ndim=16,
            dropout=0,
            attention_dropout=0,
            hidden_dropout=0,
            chunk_size=10,
            truncation=10,
            rel_pos_bias='simple',
            max_positions=30,
            activation='silu',
            attention_activation='laplace',
            bidirectional=True,
            norm_type='batchnorm',
            prenorm=True,
            feature_dropout=0,
        )

    def build_nffn_layer(self, embed_dim):
        return NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=120,
            dropout=0,
            hidden_dropout=0,
            activation='silu',
            norm_type='batchnorm',
            prenorm=True,
            feature_dropout=0,
        )

    def call(self, x, encoder_padding_mask=None,training = True):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x, _ = self.mega_layer(x, encoder_padding_mask,training = training)
        if self.nffn is not None:
            x = self.nffn(x)

        return x


