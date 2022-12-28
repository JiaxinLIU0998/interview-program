# -*- coding: utf-8 -*-
import numpy as np
import random
import pickle
import math
import os
import logging
from functools import partial
from scipy import special as ss
from einops import rearrange, repeat
import opt_einsum as oe
import tensorflow as tf
import tensorflow_probability as tfp

contract = oe.contract
contract_expression = oe.contract_expression



''' Standalone CSDI + S4 imputer for random missing, non-random missing and black-out missing.
The notebook contains CSDI and S4 functions and utilities. However the imputer is located in the last Class of
the notebook, please see more documentation of use there. Additional at this file can be added for CUDA multiplication 
the cauchy kernel.'''


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
        super(GLU, self).__init__()
        self.dim = dim
        
    def call(self, x):
        nc = x.shape[self.dim]
        assert nc % 2 == 0, 'channels dont divide 2!'
        return tf.gather(x, indices=[i for i in range(int(nc/2))], axis=self.dim) * tf.sigmoid(tf.gather(x, indices=[i for i in range(int(nc/2),nc)], axis=self.dim))



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


    def _setup_state(self):
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

