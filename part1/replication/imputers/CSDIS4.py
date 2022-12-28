import numpy as np
import random
import pickle
import math
import argparse
import datetime
import json
import os
from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import special as ss
from einops import rearrange, repeat
from imputers.transformerencoder import EncoderLayer as transformerencoder
import opt_einsum as oe
contract = oe.contract
contract_expression = oe.contract_expression
os.environ["CUDA_VISIBLE_DEVICES"] = "3"




''' Standalone CSDI + S4Model imputer. The notebook contains CSDI and S4 functions and utilities. 
However the imputer is located in the last class of the notebook, please see more documentation of use there.'''



def eigen(tensor):
    """Compute the eigenvalues and right eigenvectors."""
    c=np.linalg.eig(tensor.numpy())
    return tf.convert_to_tensor(c[0]),tf.convert_to_tensor(c[1])


def _c2r(x):
    """ Convert a complex tensor to real tensor """
    return tf.stack([tf.math.real(x, name=None) ,tf.math.imag(x, name=None)], axis=-1, name='stack')


def _r2c(x):
    """ Convert a real tensor to complex tensor """
    tran = tf.transpose(x)
    return tf.transpose(tf.complex(tran[0], tran[1]))

_conj = lambda x: tf.concat([x, tf.math.conj(x)], axis = -1, name='concat')
_resolve_conj = lambda x: tf.math.conj(x)



""" simple components """

class GLU(tf.keras.layers.Layer):
    """implementation of GLU activation function in tensorflow"""
    def __init__(self,dim=-1):
        super().__init__()
        self.dim = dim
        
        
    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.math.sigmoid(gate)
        x = tf.math.multiply(out, gate)
        return x
        

def Activation(activation=None, dim=-1):
    """ Return the required avtivation function """
    
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return tf.identity()
    elif activation == 'tanh':
        return tf.keras.layers.Activation(tf.tanh)
    elif activation == 'relu':
        return tf.keras.layers.Activation(tf.nn.relu)
    elif activation == 'gelu':
        return tf.keras.layers.Activation(tf.keras.activations.gelu)
    elif activation in ['swish', 'silu']:
        return tf.keras.layers.Activation(tf.nn.silu)
    elif activation == 'glu':
        return GLU(dim=dim) 
    elif activation == 'sigmoid':
        return tf.keras.layers.Activation(tf.Sigmoid)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation=None):
    """ Return the required initializer """
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(tf.keras.initializers.HeUniform(),nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(tf.keras.initializers.HeNormal(),nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = tf.keras.initializers.GlorotNormal()
    elif name == 'zero':
        initializer = partial(tf.keras.initializers.Constant(),val=0)
    elif name == 'one':
        initializer = partial(tf.keras.initializers.Constant(),val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer



def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a tf.keras.layers.Dense with control over axes order, initialization, and activation """

    # Construct core module
    if activation == 'glu': d_output *= 2
        
    if transposed:
        if bias:
            bound = 1 / math.sqrt(d_input)
            linear = tf.keras.layers.Dense(d_output,kernel_initializer=tf.keras.initializers.HeUniform(),
    bias_initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))
            
        else:
            linear = tf.keras.layers.Dense(d_output,kernel_initializer=tf.keras.initializers.HeUniform(),
    bias_initializer=tf.keras.initializers.Zeros())
         # Initialize weight
        if initializer is not None:
            get_initializer(initializer, activation)(linear.weight)

        # Initialize bias
        if bias and zero_bias_init:
            initializer = tf.keras.initializers.Zeros()
            initializer(linear.bias)

        # Weight norm
        if weight_norm:
            linear = tfp.layers.weight_norm.WeightNorm(linear)

        if activate and activation is not None:
            activation = Activation(activation, dim=-1)
            linear = [linear,activation]
            
    else:
        linear = tf.keras.layers.Dense(d_output,use_bias=bias,kernel_initializer=get_initializer(initializer, activation),
    bias_initializer=tf.keras.initializers.Zeros())

        if weight_norm:
            linear = tfp.layers.weight_norm.WeightNorm(linear)

        if activate and activation is not None:
            activation = Activation(activation, dim=-1)
            linear = [linear,activation]
   
    return linear



class SSKernelNPLR(tf.keras.Model):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The call pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """
    def power(self,L, A, v=None):
        """ Compute A^L and the scan sum_i A^i v_i

        A: (..., N, N)
        v: (..., N, L)
        """
        I = tf.cast(tf.eye(A.shape[-1]), dtype = A.dtype)
        powers = [A]
        l = 1
        while True:
            if L % 2 == 1: I = powers[-1] @ I
            L //= 2
            if L == 0: break
            l *= 2
            powers.append(powers[-1] @ powers[-1])
        if v is None: return I

        # Invariants:
        # powers[-1] := A^l
        # l := largest po2 at most L

        k = v.get_shape().as_list()[-1] - l
        v_ = powers.pop() @ v[..., l:]
        v = v[..., :l]
        v[..., :k] = v[..., :k] + v_

        # Handle reduction for power of 2
        while v.get_shape().as_list()[-1] > 1:
            v = rearrange(v, '... (z l) -> ... z l', z=2)
            v = v[..., 0, :] + powers.pop() @ v[..., 1, :]

        return I, tf.squeeze(v, axis=-1)

    def _setup_C(self, double_length=False):
        """ Construct C~ from C
        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        C_ = _conj(C)
       
        prod = contract("h m n, c h n -> c h m",  tf.transpose(self.power(self.L, self.dA),perm = [0,2,1]), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C = tf.Variable(_c2r(C_),trainable=True,name = "C",shape=_c2r(C_).shape)
        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, cache=True)
    
    def _omega(self, L, dtype,  cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega = tf.convert_to_tensor(np.exp(-2j * np.pi / (L)), dtype=dtype)
        #omega = tf.math.exp(-2j * tf.constant(math.pi,dtype=dtype) / tf.cast(L,dtype=dtype))  # \omega_{2L}
        omega = omega ** tf.cast(tf.range(start = 0, limit = L // 2 + 1, delta = 1) ,dtype=tf.complex64) 
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.omega = tf.Variable(_c2r(omega),trainable=False,name='omega')
            self.z = tf.Variable(_c2r(z),trainable=False,name = 'z')
        return omega, z
    
    def cauchy_slow(self,v, z, w):
        """ Cauchy kernel """
        """
        v, w: (..., N)
        z: (..., L)
        returns: (..., L)
        """
        cauchy_matrix = tf.expand_dims(v,-1) / (tf.expand_dims(z,-2) - tf.expand_dims(w,-1)) # (... N L)
        return tf.math.reduce_sum(cauchy_matrix, axis=-2)

    
    def __init__(
        self,
        L, w, P, B, C, log_dt,
        trainable=None,
        length_correction=True,
       
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        trainable: toggle which of the parameters is trainable
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """
        
        super().__init__()

        # Rank of low-rank correction
        self.rank = P.shape[-2]

        assert w.shape[-1] == P.shape[-1] == B.shape[-1] == C.shape[-1]
        self.H = log_dt.shape[-1]
        self.N = w.shape[-1]
        H = self.H
        # Broadcast everything to correct shapes
        
        C = tf.broadcast_to(C,tf.broadcast_static_shape(C.shape, tf.TensorShape([1, self.H, self.N])))
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, cache=True)

        # Register parameters
        self.C = tf.Variable(_c2r(_resolve_conj(C)),trainable=True,name = "C",shape=_c2r(_resolve_conj(C)).shape)
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True


        self.log_dt = tf.Variable(log_dt,trainable=trainable.get('dt', train),name = 'log_dt',shape=log_dt.shape)

        self.B  = tf.Variable(_c2r(B),trainable=trainable.get('B', train),name = 'B',shape=_c2r(B).shape)

        self.P = tf.Variable(_c2r(P),trainable=trainable.get('P', train),name ='P',shape=_c2r(P).shape)
        self.w = tf.Variable(_c2r(w),trainable=trainable.get('A', 0),name = "w",shape=_c2r(w).shape)
       
        P_clone = tf.identity(P)

        self.Q = tf.Variable(_c2r(_resolve_conj(P_clone)),trainable=trainable.get('P', train),name = "Q", shape = _c2r(_resolve_conj(P_clone)).shape)
        
        if length_correction:
            self._setup_C()

    def _w(self):
        """ Get the internal w (diagonal) parameter """
        w = _r2c(self.w)  # (..., N)
           
        return w

    def call(self, state=None, rate=1, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        #L = tf.cast(L,dtype = tf.float32)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
     
        while rate * L > self.L:
            self.double_length()
   
        dt = tf.math.exp(self.log_dt) * rate
        B = _r2c(self.B)
        
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()
        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype,  cache=False)

        # Augment B
        if state is not None:
            s = _conj(state) if state.shape[-1] == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
           
            s = s /tf.expand_dims(dt, -1) + sA / 2
            s = s[..., :self.N]

            B = tf.concat([s, B], axis=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * tf.cast(tf.expand_dims(dt,axis = -1),dtype = tf.complex64)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = tf.concat([B, P], axis=-3) # (s+1+r, H, N)
        C = tf.concat([C, Q], axis=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = tf.expand_dims(B,-3) * tf.expand_dims(C,-4)  # (s+1+r, c+r, H, N)
       
        # Calculate resolvent at omega
        r = self.cauchy_slow(v, z, w)
        r = r * tf.cast(dt[None, None, :, None],dtype = tf.complex64)  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = tf.linalg.inv(tf.math.eye(self.rank) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - tf.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = tf.signal.irfft(k_f)  # (S+1, C, H, L)

       
        k = tf.gather(k,[i for i in range(int(L))],axis=3)

        if state is not None:
            k_state = tf.gather(k,[i for i in range(k.shape[0]-1)],axis = 0)
        else:
            k_state = None
            
        k_B = k[-1, :, :, :] # (C H L)
        return k_B, k_state


    def double_length(self):
        self._setup_C(double_length=True)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
 
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = tf.math.exp(self.log_dt)

        D = tf.math.reciprocal((tf.cast(2.0/ tf.expand_dims(dt,-1) ,dtype=tf.complex64)  - w))  # (H, N)
        R = (tf.eye(self.rank, dtype=w.dtype) + tf.cast(tf.math.real(2*contract('r h n, h n, s h n -> h r s', Q, D, P)),dtype=tf.complex64)) # (H r r)

        Q_D = rearrange(Q*D, 'r h n -> h r n')

        R = tf.linalg.solve(tf.cast(R, dtype = Q_D.dtype), Q_D) # (H r N)

        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": (tf.cast(2.0/ tf.expand_dims(dt,-1) ,dtype=tf.complex64)+ w), # (H N)
        } 
            

    def _step_state_linear(self, u=None, state=None,gpu=True):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.
        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
 
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = tf.zeros(self.H, dtype=C.dtype)
        if state is None: # Special case used to find dB
            state = tf.zeros([self.H, self.N], dtype=C.dtype)

        step_params = self.step_params.copy()
        if state.shape[-1] == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.shape[-1] == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: tf.convert_to_tensor(contract('r h n, r h m, ... h m -> ... h n', p.numpy(), x.numpy(), y.numpy())) # inner outer product
           
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * tf.expand_dims(u,-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state


    def _setup_state(self,gpu=True):
        """ Construct dA and dB for discretized state equation """
        # Construct dA and dB by using the stepping
        
        self._setup_linear()
        
        state = tf.expand_dims(tf.eye(2*self.N, dtype=(_r2c(self.C)).dtype),-2) # (N 1 N)
        self.dA = rearrange(self._step_state_linear(state=state), "n h m -> h m n") # (H N N)
        
        u = tf.ones(self.H,dtype =(_r2c(self.C)).dtype)
        self.dB = rearrange(_conj(self._step_state_linear(u=u)), '1 h n -> h n') # (H N)

        


class HippoSSKernel(tf.keras.Model):
  
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments."""
    
    def transition(self, N, **measure_args):
        """ A, B transition matrices 
        """
        q = tf.range(start=0,limit = N, dtype=tf.float32)
        col, row = tf.meshgrid(q, q)
        r = 2 * q + 1
        M = -(tf.where(row >= col, r, 0) - tf.linalg.diag(q))
        T = tf.math.sqrt(tf.linalg.diag(2 * q + 1))
        A = T @ M @ tf.linalg.inv(T)
        B = tf.linalg.diag_part(T)[:, None]
     
        return A, B

    def rank_correction(self, N, rank=1, dtype=tf.float32):
        """ Return low-rank matrix L such that A + L is normal """
        assert rank >= 1
        P = tf.expand_dims(tf.math.sqrt(.5+tf.range(start=0, limit=N, delta=1, dtype=dtype)),0) # (1 N)
        d = P.shape[0] 
        if rank > d:
            P = tf.concat([P, tf.zeros((rank-d, N), dtype=dtype)], axis=0) # (rank N)
        return P

    def nplr(self, N, rank=1, dtype=tf.float32):
        """ Return w, p, q, V, B such that
        (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
        i.e. A = V[w - p q^*]V^*, B = V B
        """
        assert dtype == tf.float32 or tf.complex64

        A, B = self.transition(N)
        B = B[:, 0] 

        P = self.rank_correction(N, rank=rank, dtype=dtype)
        AP = A + tf.math.reduce_sum(tf.expand_dims(P, -2)*tf.expand_dims(P, -1), axis=-3)
        w, V = eigen(AP)

        # Only keep one of the conjugate pairs
        w = w[..., 0::2]
        V = V[..., 0::2]

        V_inv = tf.transpose(tf.math.conj(V),perm = [1,0])
        B = tf.einsum('ij, j -> i', V_inv, tf.cast(B, dtype = V.dtype)) # V^* B
        P = tf.einsum('ij, ...j -> ...i', V_inv, tf.cast(P, dtype = V.dtype)) # V^* P

        return w, P, B, V
    

    def __init__(
        self,
        H,
        N=64,
        L=1,
        rank=1,
        channels=1, 
        dt_min=0.001,
        dt_max=0.1,
        trainable=None, 
        length_correction=True, 
        resample=False, 
        ):
        
        """Initializer.
         Args:
            H: 2*C
            N: the dimension of the state, also denoted by N
            L: the maximum sequence length
            rank: used for HIPPO
            channels: 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min: used to learn the step size log_dt
            dt_max: used to learn the step size log_dt
            trainable: Dictionary of options to train various HiPPO parameters
            lr:  Dictionary of HiPPO parameters's learning rate
            length_correction: bool, Multiply by I-A|^L after initialization; can be turned off for initialization speed
            resample: bool, If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this   case, as it assumes that L is the true underlying length of the continuous signal
       
        """
        
        super().__init__()
        
        
        self.N = N
        self.H = H
        L = L or 1
        dtype =  tf.float32
        cdtype = tf.complex64
        self.rate = None if resample else 1
        self.channels = channels

        # Generate dt
        log_dt =  tf.random.uniform([self.H],minval=0,maxval=1,dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        w, p, B, _ = self.nplr(self.N, rank, dtype=dtype)

        C =  tf.complex(tf.random.uniform([channels, self.H, self.N // 2],minval=0,maxval=1),tf.random.uniform([channels, self.H, self.N // 2],minval=0,maxval=1))
       
        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            trainable=trainable,
            length_correction=length_correction,
        )

    
    def call(self, u = None):
        """Pass to S4.
        Args:
            inputs: L, sequence length
        Returns:
            outputs: tf.Tensor, [2*C, H, L]
        """
        
        L = u.shape[-1]
        k, _ = self.kernel(state=None,rate=self.rate, L=L)
        return tf.cast(k, dtype=tf.float32, name=None)

    


class S4(tf.keras.Model):
    """Receive the kernel constructed by HippoSSKernel and generate output y = y + Du """

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
           
            # SSM Kernel arguments
            **kernel_args,
        ):
        """Initializer.
        Args:
            d_state: the dimension of the state, also denoted by N
            l_max: the maximum sequence length, also denoted by L
              if this is not known at model creation, set l_max=1
            channels: can be interpreted as a number of "heads"
            bidirectional: bidirectional
            dropout: standard dropout argument
            transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

            Other options are all experimental and should not need to be configured
        """
        
        super().__init__()
      
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed


        self.D = tf.Variable(tf.random.uniform([channels, self.h],minval=0,maxval=1),trainable=True)

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        
        dropout_fn = tf.keras.layers.SpatialDropout2D if self.transposed else tf.keras.layers.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else tf.identity

        
        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )
        

    def call(self, u, **kwargs): # absorbs return_output and transformer src mask
        """Pass to S4.
        Args:
            inputs: tf.Tensor,u: [B, 2*C, L]  if self.transposed else [B, L, 2*C]
        Returns:
            outputs: tf.Tensor, same shape as u.
        """
       
        if not self.transposed: u = tf.transpose(u,perm = [0,2,1])
        
        L = u.shape[-1]
        # Compute SS Kernel
        
        k = self.kernel(u) # (C H L) (B C H L)
        
        # Convolution
        if self.bidirectional:
            kall = rearrange(k, '(s c) h l -> s c h l', s=2)
            k0 = kall[0]
            k1 = kall[1]
            k = tf.pad(k0, [[0,0],[0,0],[0, L]], "CONSTANT")  +  tf.pad(tf.reverse(k1,[-1]), [[0,0],[0,0],[L,0]], "CONSTANT")
           
        k_f = tf.signal.rfft(k, fft_length=tf.convert_to_tensor([2*L])) # (C H L)
        u_f = tf.signal.rfft(u, fft_length=tf.convert_to_tensor([2*L]))
        y_f = contract('bhl,chl->bchl', u_f, k_f)  # (B C H L)
        y = tf.signal.irfft(y_f, fft_length=tf.convert_to_tensor([2*L]))[..., :L]
       
        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))
        
        if not self.transposed: y = tf.transpose(y,perm = [0,2,1])
            
        if self.transposed: y = tf.transpose(y,[0,2,1])
            
        if isinstance(self.output_linear,list):
            for module in self.output_linear:
                y = module(y)
        else:
            y = self.output_linear(y)
        if self.transposed: y = tf.transpose(y,[0,2,1])
            
        return y, None



class S4Layer(tf.keras.Model):
    """A single S4layer used in residual block """
    
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        """Initializer.
        Args:
            hyperparameter defined by 'wavenet config'
        """
        super().__init__()
        self.s4_layer  = S4(d_model=features, 
                            d_state=N, 
                            l_max=lmax, 
                            bidirectional=bidirectional)
        
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-2) if layer_norm else tf.identity
        self.dropout = tf.keras.layers.SpatialDropout2D(dropout) if dropout>0 else tf.identity
    

    def call(self, x):
        """Pass to S4Layer.
        Args:
            inputs: tf.Tensor, [B, 2*C, L], output tensor
        Returns:
            outputs: tf.Tensor, [B, 2*C, L], output tensor.
        """
        xout, _ = self.s4_layer(x) #batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x # skip connection   # batch, feature, seq
        return self.norm_layer(xout)





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
    return  2 * tf.math.reduce_sum(tf.math.abs((forecast - target) * eval_points * ((target.numpy() <= forecast.numpy()) * 1.0 - q)))



def calc_denominator(target, eval_points):
    return tf.math.reduce_sum(tf.math.abs(target * eval_points))


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
                    lmax=config["lmax"],
                    
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
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,lmax):
        super().__init__()
        
        self.diffusion_projection = tf.keras.layers.Dense(channels)
        
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.time_layer = S4Layer(features=channels, lmax = lmax) 
        self.feature_layer = transformerencoder(h=nheads, d_k=64, d_v=64, d_model=64, d_ff=2048, rate=0)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
       
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]),perm = [0, 2, 1, 3]),[B * K, channel, L])
        #y = tf.transpose(self.time_layer(tf.transpose(y,perm = [2, 0, 1])),perm = [1, 2, 0])
        y = self.time_layer(y)
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
              lmax = 248,
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
        config['diffusion']['lmax'] = lmax
        
        config['model'] = {} 
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking
        
        print(json.dumps(config, indent=4))
        
        if not os.path.exists(path_save):
            os.makedirs(path_save) 
            

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
        
        
  

