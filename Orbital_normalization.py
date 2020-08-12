"""
This script reproduces Figs 1 and 2 from the following paper.

Dwyer, Christian. "Multislice theory of fast electron scattering
incorporating atomic inner-shell ionization." Ultramicroscopy 104.2
(2005): 141-151.

This is to test that all constants are correct.
"""


import pyms
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Z = 14
    bound_config = "1s2 2s2 2p6 3s2 3p2"
    excited_config = "1s2 2s2 2p6 3s2 3p2"
    n = 1
    ell = 0
    ylim = [-0.28, 0.41]
    # ylim = [-1.2,1.2]
    epsilon = 10
    grid = np.linspace(0.0, 1.0)
    bound = pyms.orbital(Z, bound_config, n, ell)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, bound(grid), "k-")
    ax.set_ylim([-2, 3])
    ax.set_xlim([np.amin(grid), np.amax(grid)])
    ax.set_xlabel("r (a.u.)")
    ax.set_ylabel("$P_{nl}(r)$")

    ax2 = ax.twinx()
    ax2.set_ylim(ylim)
    for ell in [0, 1, 2]:
        excited = pyms.orbital(Z, excited_config, 0, ell, epsilon)
        ax2.plot(grid, excited(grid), "k--")

    plt.show(block=False)
    fig.savefig("Dwyer_Fig1_replication.pdf")
    # sys.exit()

    from mpl_toolkits.mplot3d import Axes3D  # noqa

    ncols = 1
    gsize = 256
    gridshape = [gsize, gsize]
    desired_gridshape = [gsize // 4, gsize // 4]
    eV = 1e5
    gridsize = [8, 8]
    sigma = pyms.interaction_constant(eV)

    X = pyms.utils.crop(
        np.broadcast_to(
            np.linspace(-gridsize[1] / 2, gridsize[1] / 2, num=gsize).reshape(
                (1, gsize)
            ),
            gridshape,
        ),
        desired_gridshape,
    )

    Y = pyms.utils.crop(
        np.broadcast_to(
            np.linspace(-gridsize[0] / 2, gridsize[0] / 2, num=gsize).reshape(
                (gsize, 1)
            ),
            gridshape,
        ),
        desired_gridshape,
    )
    lprime = 0
    mlprime = 0

    ZZ = 14
    ncols = 3
    fig = plt.figure(figsize=(4, 4 * ncols))
    ax = fig.add_subplot("311", projection="3d")
    lprime = 0
    excited = pyms.orbital(ZZ, excited_config, 0, lprime, epsilon)
    Z = (
        pyms.utils.crop(
            np.fft.fftshift(
                np.real(
                    pyms.transition_potential(
                        bound, excited, gridshape, gridsize, 0, mlprime, eV
                    )
                )
            ),
            desired_gridshape,
        )
        / sigma
    )
    # fig2,ax2 = plt.subplots()
    # ax2.imshow(Z)
    # plt.show(block=True)

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x")
    ax.set_zlim([0.00, 0.40])
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot("312", projection="3d")
    lprime = 1
    excited = pyms.orbital(ZZ, excited_config, 0, lprime, epsilon)
    mlprime = 0
    Z = (
        pyms.utils.crop(
            np.fft.fftshift(
                np.imag(
                    pyms.transition_potential(
                        bound, excited, gridshape, gridsize, 0, mlprime, eV
                    )
                )
            ),
            desired_gridshape,
        )
        / sigma
    )
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x")
    ax.set_zlim([0.00, 0.20])
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax = fig.add_subplot("313", projection="3d")
    mlprime = 1
    Z = (
        pyms.utils.crop(
            np.fft.fftshift(
                np.real(
                    pyms.transition_potential(
                        bound, excited, gridshape, gridsize, 0, mlprime, eV
                    )
                )
            ),
            desired_gridshape,
        )
        / sigma
    )
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x")
    ax.set_zlim([-0.30, 0.30])
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show(block=True)
    fig.savefig("Dwyer_Fig2_replication.pdf")
