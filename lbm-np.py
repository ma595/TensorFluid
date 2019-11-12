import numpy as np
import array as arr
import time as ntime

# flow around a circle 
# check is working,
# dx / dt = 1 
# this means that soundspeed = cs^2 = 1/3.
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
# velocity vectors
# 0,1,2,3,4,5,6,7,8,9
v = np.array([ [ 1, 1], [ 1, 0], [ 1, -1], [ 0, 1], [ 0, 0], [ 0, -1], [-1, 1], [-1, 0], [-1, -1] ], dtype=np.int64)
t = np.array([ 1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36], dtype=np.float64)

uLB = 0.04
Re = 120

# nx = 960 
# ny = 240

nx = 1680
ny = 720
Ly = ny - 1 
Lx = nx - 1

cx, cy, r = nx//4, ny//2, ny//9

print(nx, ny)
r = ny / 9

frequency = 1000

nu = uLB*r / Re
omega = 1. / (3 * nu + 0.5 )


def circle(x, y):
    return (x - cx)**2 + (y - cy)**2 - r*r < 0


def inivel(d, x, y):
    print((1-d) * uLB * (1 + 1e-4*np.sin(y/Ly*2*np.pi)))
    return (1-d) * uLB * (1 + 1e-4*np.sin(y/Ly*2*np.pi))

def initialise(rho_global):
    rho = np.zeros((nx, ny))
    rho[:,:] = rho_global 
    u = np.fromfunction(inivel, (2,nx,ny))
    eq = equilibrium(rho, u)
    fin = np.copy(eq)
    return rho, u, eq, fin

    # print("fin", fin)
    # print("equlib", eq)
    # print("rho", rho)
    # print("u", u)

# collision term is formulated as a relaxation to local equlibrium 

# we apply periodicity here as well - is this OK given the inflow boundary condition?
# can make use of numpy arrays here as well 
def streaming(fin):
    for i in range(9):
        fin[i, :, :] = np.roll(
                np.roll(fout[i, :, :], v[i, 0], axis=0),
                v[i,1], axis=1)
    return fin

def density(fin):
    rho[:, :] = 0
    for i in range(9):
        rho[:, :] += fin[i, :, :] 
    return rho

def velocity(fin, rho):
    u = np.zeros((2,nx,ny))
    for i in range(9):
        # eins-sum
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return u

# truncated series of the Maxwell-Boltzmann distribution 
def equilibrium(rho, u):
    usqr = 3/2 * (u[0]**2 + u[1]**2)
    eq = np.zeros((9,nx,ny))
    for i in range(9):
        vu = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        eq[i, :, :] = rho * t[i] * (1 + vu + 0.5 * vu**2 - usqr)
    return eq

def apply_bcs(fin, vel):
    col1 = np.array([0, 1, 2])
    col2 = np.array([3, 4, 5])
    col3 = np.array([6, 7, 8])
    # set rho on the left boundary

    fin[col3,-1,:] = fin[col3,-2,:]
    # print("fin", fin) 

    rho = density(fin)
    u = velocity(fin, rho)
    # print(vel)
    # print(u)
   
    # print(np.shape(vel))
    # print(np.shape(u))
    u[:,0,:] = vel[:,0,:]
    # print(vel[:,0,:])
    # summing over each col set of populations 
    # return ny * 1 column vectors 
    rho[0,:] = 1/(1-u[0,0,:]) * (np.sum(fin[col2,0,:], axis=0) \
            + 2 * np.sum(fin[col3,0,:], axis=0))
    # print("rho ", rho[0,:])

    # print(" u ", u[:,0,:])
    feq = equilibrium(rho, u) # do we pass in equlibrium or recompute?
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
    fin[col1,0,:] = feq[col1,0,:] + fin[np.flip(col3),0,:] - feq[np.flip(col3),0,:]

    return fin, feq
        
def plot(time, fin):
    rho = density(fin)
    u = velocity(fin, rho)
    if time % frequency == 0:
        plt.clf()
        plt.imshow(np.sqrt(u[0]**2 + u[1]**2).transpose(), cmap=cm.Reds)
        plt.colorbar()
        plt.savefig("output/vel.{0:03d}.png".format(time//frequency))

def output(time, fin):
    frequency = 10
    rho = density(fin)
    u = velocity(fin, rho)
    if time % frequency == 0:
        f = open("output/vel.{0:03d}.dat".format(time//frequency), "w+")
        f.write("X\tY\tvel_x\tvel_y\n")
        for i in range(nx):
            for j in range(ny):
                f.write("%d\t%d\t%f\t%f\n" % (i, j, u[0,i,j], u[1,i,j]))
            f.write("\n")
        f.close()


mask = np.fromfunction(circle, (nx, ny))
vel = np.fromfunction(inivel, (2,nx,ny))

rho, u, eq, fin = initialise(1)
rho = density(fin)
maxiter = 150
start = ntime.time()
for time in range(maxiter): 
    print("time = ", time)

    # boundary conditions
    fin, feq = apply_bcs(fin, vel)
    print(fin.dtype)
    # f = open("fin.out", "w")
    # f.write(np.array2string(fin))
    # f.close()
    # if time == 1:
    #     break
    # collision 
    fout = fin - omega*(fin - feq) 
    # fout_1 = np.copy(fout)
    # bounceback
    for i in range(9):
        fout[i, mask] = fin[8-i, mask]
    # fout_2 = np.copy(fout) 

    # diff = abs(fout_1 - fout_2)
    # print("diff", np.sum(diff))
    # streaming 
    # fout = np.copy(fin)	
    fin = streaming(fin)
    # print("np ", fin[0,0,:])
    # print("np ", fin[0,nx-1,:])

    # print("feq", feq)
    # np.set_printoptions(threshold=sys.maxsize)
    # f = open("feq.out", "w")
    # f.write(np.array2string(feq))
    # f.close()

# print(fin)
    #     print(lbm_np.fin[0,0,:])
    # plot(time, fin)

    # output(time, fin)
    if time == 1500:
        break

print("total time ", ntime.time() - start)
def checks():
    print(np.reshape(fin[:,50,20], (3, 3)))
    def print_mask():
        for i in range(nx):
            for j in range(ny):
                print(mask[i, j], end=' ')
            print('')
    print_mask()

