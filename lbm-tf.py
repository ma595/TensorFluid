import numpy as np
import tensorflow as tf
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import time as ntime

# v = tf.constant([ [ 1, 1], [ 1, 0], [ 1, -1], [ 0, 1], [ 0, 0], [ 0, -1], [-1, 1], [-1, 0], [-1,-1] ], dtype=tf.float64)
v_np = np.array([ [ 1, 1], [ 1, 0], [ 1, -1], [ 0, 1], [ 0, 0], [ 0, -1], [-1, 1], [-1, 0], [-1, -1] ])
v_x = tf.constant( [1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=tf.float64)
v_y = tf.constant( [1, 0, -1, 1, 0, -1, 1, 0, -1], dtype=tf.float64)

t = tf.constant( [1./36, 1./9, 1./36, 1./9, 4./9, 1./9, 1./36, 1./9, 1./36], dtype=tf.float64)

uLB = 0.04
Re = 120

# nx = 420
# ny = 180

nx = 1680
ny = 720

Ly = ny - 1 
Lx = nx - 1

cx, cy, r = nx//4, ny//2, ny//9

print(nx, ny)
r = ny / 9

frequency = 1000
maxiters = 1500

nu = uLB*r / Re
omega = 1. / (3 * nu + 0.5 )

def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """

    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor


def circle(x, y):
    return (x - cx)**2 + (y - cy)**2 - r*r < 0

def inivel(d, x, y):
    return (1-d) * uLB * (1 + 1e-4*np.sin(y/Ly*2*np.pi))

def initialise(rho_global):
    rho = np.zeros((nx, ny))
    rho[:,:] = rho_global 
    u = np.fromfunction(inivel, (2,nx,ny))
    eq = equilibrium(rho, u)
    fin = np.copy(eq)
    return rho, u, eq, fin

# implementation of the roll function in tf. 
def streaming(X, tfilter):
    # print(np_filter[:,:,i,:])
    X = tf.transpose(X, perm=[1,2,0])
    X = tf.expand_dims(X, axis=0)
    X = periodic_padding_flexible(X, 1, 1)
    X = periodic_padding_flexible(X, 2, 1)
    # print("tf_filter, shape", tfilter.shape)
    X = tf.nn.conv2d(X, tfilter, strides=[1,1,1,1], padding="SAME")
    return tf.transpose(tf.squeeze(tf.slice(X,[0,1,1,0],[1,nx,ny,9])), perm=[2,0,1])

def streaming_roll(X):
    # return tf.roll() 
    pass
    # for i in range(9):

def density(fin):
    # rho_new = tf.get_variable("rho", initializer=tf.zeros(nx,ny), dtype=tf.float64)
    rho_new = tf.reduce_sum(fin, axis=0)
    return rho_new

def velocity(fin_cp, rho_cp):
    # u = tf.get_variable("u", initializer=tf.zeros((2,nx,ny),dtype=np.float64), dtype=tf.float64)
    # u_new = tf.einsum('ik,klm->ilm', tf.transpose(v), fin_cp)  # this is giving rise to numerical error?
    u_x = tf.einsum('k,klm->lm', v_x, fin_cp)
    u_y = tf.einsum('k,klm->lm', v_y, fin_cp)
    u_new_1 = tf.stack([u_x, u_y], axis=0)
    u_new_1 /= rho_cp
    return u_new_1

ones_1 = tf.ones([9], dtype=tf.float64)
ones_3 = tf.ones([9,nx,ny], dtype=tf.float64)
tdiag = tf.linalg.diag(t)

def equilibrium(rho, u_var):
    usqr = 3/2 * (u_var[0]**2 + u_var[1]**2)
    usqr_9 = tf.tensordot(ones_1, usqr, axes=0)
    # print(usqr_9.eval().shape)
    vu = 3 * (tf.tensordot(v_x, u_var[0], axes=0) + tf.tensordot(v_y, u_var[1], axes=0)) 
    rho_9 = tf.stack([rho] * 9)
    eq = tf.einsum('ik,klm->ilm', tdiag, tf.math.multiply(rho_9,  (ones_3 + vu + 0.5 * vu*vu - usqr_9)))
    return eq

def create_bc_mask_1():
    zeros = tf.zeros([3, nx-2, ny], dtype=tf.int32)
    ones = tf.ones([3, 1, ny], dtype=tf.int32)
    zeros_ones = tf.concat([zeros, ones], 1)
    zeros_rest = tf.zeros([3, nx-1, ny], dtype=tf.int32)
    mask = tf.concat([zeros_rest, zeros_rest, zeros_ones], 0)
    mask = tf.cast(mask, dtype=tf.bool)
    return mask

def create_bc_mask_2():
    zeros = tf.zeros([2, nx-1, ny], dtype=tf.int32)
    ones = tf.ones([2, 1, ny], dtype=tf.int32)
    mask = tf.concat([ones,zeros],1)
    mask = tf.cast(mask, dtype=tf.bool)
    return mask

mask_bc_1 = create_bc_mask_1()
mask_bc_2 = create_bc_mask_2()
ones = tf.ones([ny], dtype=tf.float64)

def apply_bcs(fin, vel):
    # apply transmissive boundary conditions at x=nx
    # we lose the first bc doing this
    # append the original x = 0 to this
    # fin[col3,-1,:] = fin[col3,-2,:]
    # these should probably all be constant     
    fin_new = tf.where(mask_bc_1, fin[:,0:nx-1, 0:ny], fin[:,1:nx, 0:ny])
    fin_new = tf.concat([fin[:, 0:1, :], fin_new], 1)

    # update density and velocity 
    rho = density(fin_new)
    u = velocity(fin_new, rho)
   
    # set velocity at x=0 to initial velocity 
    # u[:,0,:] = vel[:,0,:]
    u_new = tf.where(mask_bc_2, vel, u)

    # rho[0,:] = 1/(1-u[0,0,:]) * (np.sum(fin[col2,0,:], axis=0) \
    #         + 2 * np.sum(fin[col3,0,:], axis=0))
    # correction to rho at x = 0
    rho_reduce_tf = tf.reduce_sum(fin_new[3:6,0,:],0) + 2*tf.reduce_sum(fin_new[6:9,0,:],0)
    ones_minus_vel = ones - u_new[0,0,:]
    rho_bc = tf.math.reciprocal(ones_minus_vel) * rho_reduce_tf
    new_rho = tf.concat([[rho_bc], rho[1:nx,:]],0)

    feq = equilibrium(new_rho, u_new)
    # correction to fin populations at x = 0
    # the correction only applies to 0,1,2
    fin_correction = feq[0:3,0,:] + fin_new[8:5:-1,0,:] - feq[8:5:-1,0,:]
    fin_correction = tf.concat([fin_correction, fin_new[3:9,0,:]],0)
    fin_correction = tf.expand_dims(fin_correction, 1)
    fin_new = tf.concat([fin_correction, fin_new[:,1:nx,:]],1)
    return fin_new, feq

mask_circle = np.fromfunction(circle, (nx, ny))
mask_circle_9 = tf.stack([mask_circle] * 9)
def bounceback(fin, fout):
    # fout = tf.where(mask_circle_9, tf.reverse(fin, [0]), fin)
    fout = tf.where(mask_circle_9, tf.reverse(fin, [0]), fout)
    return fout


# create filter for streaming 
np_filter = np.zeros((3,3,9,9), np.float64)
# print("np_filter ", np_filter.shape)
for i in range(9):
    # np_filter[-1*v_np[i][0] + 1, -1*v_np[i][1] + 1, i, i] = 1
    np_filter[-1*v_np[i][0] + 1, -1*v_np[i][1] + 1, i, i] = 1
tfilter = tf.convert_to_tensor(np_filter)
print(tfilter.shape)

# fin = tf.get_variable("fin", initializer=fin)
# XV = tf.get_variable("vnp", initializer=X)
# rho, u, eq, fin = initialise(1)
# rho = density(fin)
# fin = np.zeros((1,nx,ny,9)) 
# fin = initialise(fin)
# fin = np.pad(fin, ((0,0),(1,1),(1,1),(0,0)),"wrap") 

def density_np(fin):
    rho_np = np.zeros((nx,ny))
    for i in range(9):
        rho_np += fin[i] 
    return rho_np

def velocity_np(fin, rho):
    u_np = np.zeros((2,nx,ny))
    for i in range(9):
        # eins-sum
        u_np[0] += v_np[i, 0] * fin[i, :, :]
        u_np[1] += v_np[i, 1] * fin[i, :, :]
    u_np /= rho
    return u_np

def plot_np(time, fin):
    rho_np = density_np(fin)
    u_np = velocity_np(fin, rho_np)
    if time % frequency == 0:
        plt.clf()
        plt.imshow(np.sqrt(u_np[0]**2 + u_np[1]**2).transpose(), cmap=cm.Reds)
        plt.colorbar()
        plt.savefig("output/vel.{0:03d}.png".format(time//frequency))


def plot(time, fin):
    rho = density(fin)
    u = velocity(fin, rho)
    print("do plotting")
    plt.clf()
    u = u.eval()
    plt.imshow(np.sqrt(u[0]**2 + u[1]**2).transpose(), cmap=cm.Reds)
    plt.colorbar()
    plt.savefig("output/vel.{0:03d}.png".format(time//frequency))

vel = np.fromfunction(inivel, (2,nx,ny))
vel = tf.Variable(vel)
u = tf.identity(vel) 
rho = tf.Variable(np.ones((nx,ny)))
fin = tf.Variable(tf.zeros([9,nx,ny], dtype=tf.float64))
fout = tf.Variable(tf.zeros([9,nx,ny], dtype=tf.float64))
feq = tf.Variable(tf.zeros([9,nx,ny], dtype=tf.float64))

@tf.function
def do_loop():
    fin_bc, feq_bc = apply_bcs(fin, vel)
    fin.assign(fin_bc)
    feq.assign(feq_bc)

    # do collision 
    fout.assign(fin - omega*(fin - feq))

    # do bounceback
    fout.assign(bounceback(fin, fout))

    # do streaming
    fin.assign(streaming(fout, tfilter))

dirName = "output"
try:
    # create target directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

start_total = ntime.time()
start = ntime.time()

@tf.function
def run_lbm():    
    fin.assign(equilibrium(rho,u))
    feq.assign(equilibrium(rho,u))
    start = ntime.time()

    for t in range(maxiters):
        do_loop()
        print(t)
        if t % frequency == 0:
            plot_np(t, fin)

    print("total time while loop ", ntime.time() - start)

    # profiling 
    # writer = tf.summary.FileWriter("logdir/lbm", sess.graph)
    # writer.close()
run_lbm()
