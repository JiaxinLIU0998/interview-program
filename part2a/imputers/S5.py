from functools import partial
import jax
import jax.numpy as np
from jax import random
from jax.numpy.linalg import eigh
import tensorflow as tf
from jax.scipy.linalg import block_diag
from jax.nn.initializers import lecun_normal, normal
import tensorflow_probability as tfp


################################## init ##################################
def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig




def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng,shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    
    B = init_fun(rng,shape)
    
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)




################################## ssm ##############################################   
    
# Discretization functions
def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = tf.ones(Lambda.shape[0])
    Lambda_bar = tf.math.exp(Lambda * tf.cast(Delta,tf.complex64))
    B_bar = (1/Lambda * (Lambda_bar-tf.cast(Identity,tf.complex64)))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * tf.cast(tf.ones((input_sequence.shape[0],Lambda_bar.shape[0])),tf.complex64)

    Bu_elements = tf.vectorized_map(lambda u: B_bar @ u, tf.expand_dims(tf.cast(input_sequence,tf.complex64),-1))
    Bu_elements = tf.squeeze(Bu_elements,-1)
    _, xs = tfp.math.scan_associative(binary_operator, (Lambda_elements, Bu_elements))


    if bidirectional:
        _, xs2 = tfp.math.scan_associative(fn=binary_operator, elems=(tf.reverse(Lambda_elements,[0]), tf.reverse(Bu_elements,[0])))
        xs2 = tf.reverse(xs2,[0])
      
        xs = tf.concat((xs, xs2), axis=-1)

    if conj_sym:
        return tf.math.real(tf.squeeze(tf.vectorized_map(lambda x: 2*(C_tilde @ x), tf.expand_dims(xs,-1)),-1))
        
    else:
        return tf.math.real(tf.squeeze(tf.vectorized_map(lambda x: (C_tilde @ x), tf.expand_dims(xs,-1)),-1))
        


class S5SSM(tf.keras.Model):
 

    """ The S5 SSM
       
    """
    
    def __init__(self,Lambda_re_init, Lambda_im_init, V, Vinv, H, P, C_init, discretization, dt_min, dt_max, conj_sym = True, clip_eigs = False,bidirectional=True, step_rescale=1.0):
        """Initializer.
          Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        
        self.Lambda_re_init = Lambda_re_init
        self.Lambda_im_init = Lambda_im_init
        self.V = V
        self.Vinv = Vinv
        self.H = H
        self.P = P
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale
        

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P
            
 
        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = tf.Variable(self.Lambda_re_init, trainable=True, name = "Lambda_re" )
        self.Lambda_im = tf.Variable(self.Lambda_im_init, trainable=True, name = "Lambda_im" )
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda =tf.complex(self.Lambda_re, self.Lambda_im)

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        
        
        self.B = tf.Variable(init_VinvB(B_init,jax.random.PRNGKey(0), B_shape,self.Vinv), trainable=True, name = "B" )
                            
        B_tilde =tf.complex(self.B[..., 0], self.B[..., 1])             

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = tf.Variable(C_init, trainable=True, name = "C",shape= (self.H, 2 * self.P, 2) )
                self.C_tilde =tf.complex( C[..., 0], C[..., 1])

            else:
                C = tf.Variable(C_init, trainable=True, name = "C",shape= (self.H, self.P, 2))
                self.C_tilde =tf.complex( C[..., 0], C[..., 1])


        else:
            if self.bidirectional:
                self.C1 = tf.Variable(init_CV(C_init, jax.random.PRNGKey(1),C_shape, self.V), trainable=True, name = "C1",shape=init_CV(C_init, jax.random.PRNGKey(1),C_shape, self.V).shape)
           
                self.C2 = tf.Variable(init_CV(C_init,jax.random.PRNGKey(2), C_shape, self.V), trainable=True, name = "C2", shape=init_CV(C_init,jax.random.PRNGKey(2), C_shape, self.V).shape)
           
                C1 =tf.complex(self.C1[..., 0], self.C1[..., 1])
                C2 =tf.complex(self.C2[..., 0], self.C2[..., 1])
                self.C_tilde = tf.concat((C1, C2), axis=-1)

            else:
                self.C = tf.Variable(init_CV(C_init, jax.random.PRNGKey(3),C_shape, self.V), trainable=True, name = "C",shape=init_CV(C_init, jax.random.PRNGKey(3),C_shape, self.V).shape)
               
                self.C_tilde =tf.complex( self.C[..., 0],self.C[..., 1])
        # Initialize feedthrough (D) matrix
        d_init = normal(stddev=1.0)
        self.D = tf.Variable(d_init(jax.random.PRNGKey(4),shape=(self.H,1)), trainable=True, name = "D",shape=(self.H,1))

        # Initialize learnable discretization timescale value
        self.log_step = tf.Variable(init_log_steps(jax.random.PRNGKey(5),(self.P, self.dt_min, self.dt_max)), trainable=True, name = "log_step")
    
        step = self.step_rescale * tf.math.exp(self.log_step[:, 0])

        # Discretize
        
        self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
       
    def call(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_sequence,
                       self.conj_sym,
                       self.bidirectional)

        # Add feedthrough matrix output Du;
        Du = tf.squeeze(tf.vectorized_map(lambda u: self.D * u, tf.expand_dims(input_sequence,-1)),-1)
        
        return ys + Du





############################### s5 layer ########################################################


class SequenceLayer(tf.keras.Model):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
       
    """
        
    def __init__(self, d_model, dt_min=0.001, dt_max=0.1, block_size = 32, blocks = 8, conj_sym = True, ssm_size = 256, dropout = 0.0, activation = 'gelu',training = True, prenorm = False, batchnorm = False, bn_momentum = 0.90, step_rescale = 1.0):
        """Initializer.
         Args:
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
        """
        super().__init__()
        
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

   

        
        self.seq = S5SSM(H= d_model,
                         P=ssm_size,
                         Lambda_re_init=Lambda.real,
                         Lambda_im_init=Lambda.imag,
                         V=V,
                         Vinv=Vinv,
                         C_init='trunc_standard_normal',
                         discretization="zoh",
                         dt_min=dt_min,
                         dt_max=dt_max,
                         conj_sym=conj_sym,
                         clip_eigs=False,
                         bidirectional=True,
                         step_rescale=step_rescale,
                        )

        
   
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.prenorm = prenorm
        if self.batchnorm:
   
            self.norm = tf.keras.layers.BatchNormalization(axis=-1,epsilon=1e-05,momentum=self.bn_momentum)
        else:
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-06)
            
        self.drop =  tf.keras.layers.Dropout(self.dropout)


    def call(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
            
        x = tf.vectorized_map(self.seq, x)
        
        x = self.drop(tf.keras.activations.gelu(x,approximate=False))
        
        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
    