import numpy as np
import time as ntime

import matplotlib.pyplot as plt
from matplotlib import cm
import sys

# velocity vectors
# 0,..,9
v = np.array(
    [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]],
    dtype=np.int64,
)

t = np.array(
    [
        1.0 / 36,
        1.0 / 9,
        1.0 / 36,
        1.0 / 9,
        4.0 / 9,
        1.0 / 9,
        1.0 / 36,
        1.0 / 9,
        1.0 / 36,
    ],
    dtype=np.float64,
)


def initialise(rho_global, nx, ny):
    rho = np.zeros((nx, ny))
    rho[:, :] = rho_global
    u = np.fromfunction(inivel, (2, nx, ny))
    eq = equilibrium(rho, u)
    fin = np.copy(eq)
    return rho, u, eq, fin


# we apply periodicity here as well - is this OK given the inflow boundary condition?
# can make use of numpy arrays here as well
def streaming(fin):
    """Streaming takes the system to the next time iteration, 𝑡 + 𝛿𝑡"""
    # this function warrants some explanation
    for i in range(9):
        fin[i, :, :] = np.roll(np.roll(fout[i, :, :], v[i, 0], axis=0), v[i, 1], axis=1)
    return fin


def density(fin, rho):
    rho[:, :] = 0
    for i in range(9):
        rho[:, :] += fin[i, :, :]
    return rho


def velocity(fin, rho, nx, ny):
    u = np.zeros((2, nx, ny))
    for i in range(9):
        # vectorisation
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return u


def equilibrium(rho: np.ndarray, u: np.ndarray, nx, ny):
    """Obtained as a truncated series of the Maxwell-Boltzmann distribution"""
    usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    eq = np.zeros((9, nx, ny))
    for i in range(9):
        vu = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        eq[i, :, :] = rho * t[i] * (1 + vu + 0.5 * vu**2 - usqr)
    return eq


def apply_bcs(fin, vel, nx, ny):
    """apply which boundary conditions"""
    col1 = np.array([0, 1, 2])
    col2 = np.array([3, 4, 5])
    col3 = np.array([6, 7, 8])

    # set rho on the left boundary
    fin[col3, -1, :] = fin[col3, -2, :]

    rho = density(fin)
    u = velocity(fin, rho)

    # print(np.shape(vel))
    # print(np.shape(u))
    u[:, 0, :] = vel[:, 0, :]
    # print(vel[:,0,:])
    # summing over each col set of populations
    # return ny * 1 column vectors
    rho[0, :] = (
        1
        / (1 - u[0, 0, :])
        * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3, 0, :], axis=0))
    )

    feq = equilibrium(rho, u, nx, ny)  # do we pass in equlibrium or recompute?
    # f = open("u.out", "w")
    # f.write(np.array2string(u))
    # f.close()
    # f = open("rho.out", "w")
    # f.write(np.array2string(rho))
    # f.close()
    # f = open("fin_new.out", "w")
    # f.write(np.array2string(fin))
    # f.close()
    # correction to fin populations
    fin[col1, 0, :] = (
        feq[col1, 0, :] + fin[np.flip(col3), 0, :] - feq[np.flip(col3), 0, :]
    )

    return fin, feq


# this function should probably be moved elsewhere
def plot(time, fin, frequency):
    rho = density(fin)
    u = velocity(fin, rho)
    if time % frequency == 0:
        plt.clf()
        plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap=cm.Reds)
        plt.colorbar()
        plt.savefig("output/vel.{0:03d}.png".format(time // frequency))


def output(time, fin, frequency):
    rho = density(fin)
    u = velocity(fin, rho)
    if time % frequency == 0:
        f = open("output/vel.{0:03d}.dat".format(time // frequency), "w+")
        f.write("X\tY\tvel_x\tvel_y\n")
        for i in range(nx):
            for j in range(ny):
                f.write("%d\t%d\t%f\t%f\n" % (i, j, u[0, i, j], u[1, i, j]))
            f.write("\n")
        f.close()


def loop(maxiter, fin, vel, mask, omega, output_frequency):
    vel = state.vel

    """ loop for maxiters """
    for time in range(maxiter):
        print("time = ", time)

        # apply boundary conditions
        fin, feq = apply_bcs(fin, vel, nx, ny)

        # debugging
        # f = open("fin.out", "w")
        # f.write(np.array2string(fin))
        # f.close()

        # collision
        fout = fin - omega * (fin - feq)

        #  bounceback
        for i in range(9):
            fout[i, mask] = fin[8 - i, mask]
        # fout_2 = np.copy(fout)

        # diff = abs(fout_1 - fout_2)
        # print("diff", np.sum(diff))
        # fout = np.copy(fin)

        # streaming
        fin = streaming(fin)
        # print("np ", fin[0,0,:])
        # print("np ", fin[0,nx-1,:])

        # print("feq", feq)
        # np.set_printoptions(threshold=sys.maxsize)
        # f = open("output/feq.out", "w")
        # f.write(np.array2string(feq))
        # f.close()

        # print(fin)
        #     print(lbm_np.fin[0,0,:])
        # plot(time, fin)

        # output(time, fin)

    print("total time ", ntime.time() - start)
    return fin


def obstacle_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 - r * r < 0


def inivel(d, x, y, Lx, Ly):
    # print((1 - d) * uLB * (1 + 1e-4 * np.sin(y / Ly * 2 * np.pi)))
    return (1 - d) * uLB * (1 + 1e-4 * np.sin(y / Ly * 2 * np.pi))


def solve_cylinder():
    """flow around cylinder"""

    # dx / dt = 1
    # therefore soundspeed = cs^2 = 1/3.

    # velocity at inlet
    uLB = 0.04

    # Reynolds number
    Re = 120

    # Number of cells
    nx, ny = 420, 180
    Lx = nx - 1
    Ly = ny - 1

    # viscosity
    nu = uLB * r / Re

    # set cylinder placement and dimensions
    cx, cy, r = nx // 4, ny // 2, ny / 9

    output_frequency = 1000

    # relaxation parameter
    omega = 1.0 / (3 * nu + 0.5)

    mask = np.fromfunction(obstacle_circle, (nx, ny, cx, cy, r))
    vel = np.fromfunction(inivel, (2, nx, ny, Lx, Ly))

    rho, u, eq, fin = initialise(1, nx, ny)
    rho = density(fin)
    maxiter = 150
    state = {"rho": rho, "vel": vel, "fin": fin}
    grid = {"nx": nx, "ny": ny, "mask": mask}
    loop(state, grid, maxiter, omega, output_frequency)


def checks():
    print(np.reshape(fin[:, 50, 20], (3, 3)))

    def print_mask():
        for i in range(nx):
            for j in range(ny):
                print(mask[i, j], end=" ")
            print("")

    print_mask()
