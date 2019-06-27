'''

'''
import tensorflow as tf

def layca(p, step, lr):
    '''
    Core operations of layca.
    Takes the current parameters and the step computed by an optimizer, and 
         - projects and normalizes the step such that the rotation operated on the layer's weights is controlled
         - after the step has been taken, recovers initial norms of the parameters
         
    !!! 
    only kernels are optimized. Biases and batchnorm pararmeters are left unchanged. This did not affect performance in our experiments. 
    One can decide to train them anyway (without layca operations) by changing last line to:
    return p  - lr * step
    !!!
    '''
    if 'kernel' in p.name: # only kernels are optimized when using Layca (and not biases and batchnorm parameters)
        # projecting step on tangent space of sphere -> orthogonal to the parameters p
        initial_norm = tf.norm(p)
        step = step - (  tf.reduce_sum(step * p, axis=None, keepdims=False)  )* p / initial_norm**2

        # normalizing step size (with special attention to numerical problems)
        step = tf.cond(tf.norm(step)<= 1e-7, lambda: tf.zeros_like(step), lambda: step/ (tf.norm(step)) * initial_norm)
        
        # applying step
        new_p =  p - lr * step

        # recovering norm of the parameter from before the update
        new_p = new_p / tf.norm(new_p) * initial_norm
        return new_p
    else:
        return p # - lr * step  # uncomment to train biases and batchnorm parameters (without layca)
            
