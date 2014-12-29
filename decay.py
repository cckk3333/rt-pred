# coding: utf-8

import FTRLProximal

def decay(learner,theta):
    ''' decay n_i^2 to prevent small learning rate

        we decay n_i where w_i > 0 and
        we modify the z_i so that the w_i doesn't change

        here is the formula:

        n_i  <-  theta * n_i
        z_i  <- ( (1 - sqrt(theta)) * L1 * sgn(z_i) + z_i * (L2 + (beta + sqrt(theta)) / alpha ) ) / (L2 + (beta + sqrt(n_i) / alpha))
    '''
    alpha = learner.alpha
    beta = learner.beta
    L1 = learner.L1
    L2 = learner.L2
    
    z = learner.z
    n = learner.n
    w = learner.w

    for i in w:
        sign = -1. if z[i] < 0 else 1.
        if sign * z[i] < L1:
            z[i], n[i] = ( (1 - sqrt(theta)) * sqrt(n[i]) * L1 * sign / alpha + z[i] * (L2 + (beta + sqrt(theta*n[i])) / alpha ) ) / (L2 + (beta + sqrt(n[i])) / alpha) ,  theta * n[i]
            
    


