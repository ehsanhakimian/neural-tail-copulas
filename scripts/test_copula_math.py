#!/usr/bin/env python3
"""Small deterministic tests for the main copula mathematics and file logic."""
import sys, pathlib, unittest
import numpy as np
from scipy import integrate
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import core_engine as core

class CopulaTests(unittest.TestCase):
    def test_log_density_matches_density(self):
        rng=np.random.default_rng(7)
        u=rng.uniform(.02,.98,200)
        v=rng.uniform(.02,.98,200)
        theta=rng.uniform(.2,5.0,200)
        lhs=core.clayton_log_density_np(u,v,theta)
        a=u**(-theta)+v**(-theta)-1
        ref=np.log1p(theta)-(1+theta)*(np.log(u)+np.log(v))-(2+1/theta)*np.log(a)
        self.assertTrue(np.allclose(lhs,ref,atol=1e-10))
    def test_mixture_finite(self):
        rng=np.random.default_rng(9)
        u=rng.uniform(.01,.99,(30,4))
        theta=np.full((30,3),1.5)
        weights=np.full_like(theta,.35)
        ll=core.star_loglik_np(u,theta,theta,weights)
        self.assertTrue(np.isfinite(ll).all())
    def test_copula_density_normalization(self):
        for u in (.1,.5,.9):
            integral=integrate.quad(lambda v: float(core.clayton_density_np(u,v,2.0)),1e-7,1-1e-7,epsabs=1e-7)[0]
            self.assertAlmostEqual(integral,1.0,places=4)
    def test_inverse_conditional(self):
        for u in (.15,.45,.85):
            p=.37; theta=1.8
            v=core.clayton_h_inv(p,u,theta)
            S=u**(-theta)+v**(-theta)-1
            recovered=u**(-theta-1)*S**(-1/theta-1)
            self.assertAlmostEqual(float(recovered),p,places=6)
    def test_ewma_has_no_lookahead(self):
        rng=np.random.default_rng(5)
        U=rng.uniform(.05,.95,(45,3)); score=core.ewma_gaussian_ll(U,train_end=20)
        modified=U.copy(); modified[35:]=rng.uniform(.05,.95,(10,3))
        newscore=core.ewma_gaussian_ll(modified,train_end=20)
        self.assertTrue(np.allclose(score[20:35],newscore[20:35],atol=1e-10))

if __name__=='__main__':unittest.main(verbosity=2)
