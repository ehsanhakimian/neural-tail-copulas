import unittest, numpy as np
from scipy.integrate import simpson
from four_rotation_clayton_PROSPECTIVE import four_rotation_logdensity, simplex_from_logits
class TestRotatedMixture(unittest.TestCase):
    def test_simplex(self):
        w=simplex_from_logits([1,2,-3,0]);self.assertAlmostEqual(float(w.sum()),1,12)
        self.assertTrue(np.all(w>0))
    def test_integral(self):
        # Numerical sanity check: a mixture of rotated copula densities integrates to 1.
        q=np.linspace(0.0003,0.9997,251)
        u,v=np.meshgrid(q,q)
        for weights in ([.25,.25,.25,.25],[0,1,0,0],[.8,0,0,.2]):
            f=np.exp(four_rotation_logdensity(u,v,1.2,weights))
            integral=simpson(simpson(f,x=q,axis=0),x=q)
            self.assertLess(abs(integral-1),.035)
    def test_rejects_weights(self):
        with self.assertRaises(ValueError):four_rotation_logdensity(.5,.5,1,[.2,.2,.2,.2])
if __name__=='__main__':unittest.main()
