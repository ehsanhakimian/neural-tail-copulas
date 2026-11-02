"""Four-rotation Clayton mixture: a prospective model extension, not a reported result.

The original Clayton/survival-Clayton root edges cannot describe negative
quadrant dependence. This module defines a normalized mixture of the 0, 90,
180 and 270-degree rotations. It has NOT been fitted to the external holdout.
Any performance comparison must be declared exploratory unless evaluated on
new data after finalizing the design and hyperparameters.
"""
import numpy as np
from scipy.special import logsumexp, softmax

def _clayton_logc(u, v, theta):
    """Stable Clayton density at interior probabilities and positive theta."""
    u=np.clip(np.asarray(u,dtype=float),1e-10,1-1e-10)
    v=np.clip(np.asarray(v,dtype=float),1e-10,1-1e-10)
    th=np.asarray(theta,dtype=float)
    if np.any(th<=0):raise ValueError('theta must be strictly positive')
    a=-th*np.log(u); b=-th*np.log(v)
    logq=np.logaddexp(a,b)+np.log1p(-np.exp(-np.logaddexp(a,b)))
    return np.log1p(th)-(1+th)*(np.log(u)+np.log(v))-(2+1/th)*logq

def rotated_clayton_logdensities(u, v, theta):
    """Return components ordered 0, 90, 180, 270 degrees."""
    u=np.asarray(u,dtype=float);v=np.asarray(v,dtype=float)
    return np.stack([_clayton_logc(u,v,theta),
       _clayton_logc(1-u,v,theta),
       _clayton_logc(1-u,1-v,theta),
       _clayton_logc(u,1-v,theta)],axis=-1)

def four_rotation_logdensity(u,v,theta,weights):
    """Exact log density of a valid four-component rotated copula mixture."""
    weights=np.asarray(weights,dtype=float)
    if weights.shape != (4,) or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('weights must be four nonnegative numbers summing to one')
    with np.errstate(divide='ignore'):
        return logsumexp(rotated_clayton_logdensities(u,v,theta)+np.log(weights),axis=-1)

def simplex_from_logits(logits):
    """Convert four unconstrained neural outputs into valid mixture weights."""
    logits=np.asarray(logits,dtype=float)
    if logits.shape[-1]!=4:raise ValueError('last axis must have four entries')
    return softmax(logits,axis=-1)
