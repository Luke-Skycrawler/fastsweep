import shaysweep
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from drjit.cuda import TensorXf
import fastsweep 
def show(fig, phi):
    from matplotlib import colors
    divnorm = colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    img = fig.imshow(phi.T, origin="lower", animated=True,
                        cmap='PiYG', norm=divnorm)  # , interpolation="antialiased")
    phinp = phi
    print(f"phi max = {np.max(phinp)}, min = {np.min(phinp)}")
    
def test():
    r = shaysweep.Redistance(3)
    res = 128
    phi = np.zeros((res, res, res))
    h = 1.0 / phi.shape[0]
    scale = h / 2
    # scale = 
    phi[:] = 1.0 * scale
    phi[:, : res // 2, :] = -1.0 * scale
    # r.redistance_3d(phi, h, 0)

    init = phi.copy()
    init_ = TensorXf(init)
    start_0 = time.time()
    ref = fastsweep.redistance(init_).numpy()
    end_0 = time.time()
    
    
    start = time.time()
    phi = r.redistance_3d(phi, h, 1)
    end = time.time()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    err = ref - phi
    print(f"max error = {np.max(np.abs(err))}")
    show(ax1, phi[res // 2, :, :])
    show(ax2, err[res // 2, :, :])
    plt.show()
    print(f"time = {end -start}s, ref = {end_0 - start_0}")
test()