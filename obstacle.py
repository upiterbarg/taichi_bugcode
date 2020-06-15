import taichi as ti
import taichi as tc
import os
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import imageio
import sys
import numpy as np

real = ti.f32
ti.init(arch=ti.cpu, default_fp=real)

real = ti.f32
dim = 2
gravity = 9.8
n_grid = 144
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-4
p_mass = 1
p_vol = 1
E, nu = 5e3, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) 
floor = 0.03

max_steps = 10000  # length of each simulation run in timesteps
num_iters = 201 # total number of training iterations

n_fluid_particles_x = 20
n_fluid_particles_y = 10
n_particles = 2000
n_container_particles = 0
scalar = lambda: ti.var(dt=real)
vec = lambda: ti.Vector(dim, dt=real)
mat = lambda: ti.Matrix(dim, dim, dt=real)

x, v, C, F = vec(), vec(), mat(), mat() #  position, velocity, affine velocity, deformation gradient
grid_v_in, grid_v_out, grid_m, grid_c = vec(), vec(), scalar(), scalar()  # grid momentum and velocity
Jp = scalar() # plastic deformation
particle_type = scalar() # track particle types
stuck = scalar()

loss = scalar() # track loss
learning_rate = 0.01 # set learning rate

# quantities to be learned
init_v = scalar()

# --------------- Memory Preallocation ---------------
@ti.layout
def place():
    ti.root.dense(ti.l, max_steps).dense(ti.k, n_particles).place(x, x.grad, v, C, F, Jp)
    ti.root.dense(ti.k, n_particles).place(particle_type)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_v_out, grid_m, grid_c)
    ti.root.dense(ti.l, max_steps).place(stuck)
    ti.root.place(init_v)
    ti.root.place(loss)
    ti.root.lazy_grad()

# --------------- Training Utils ---------------
def zero_vec():
  return [0.0, 0.0]


def zero_matrix():
  return [zero_vec(), zero_vec()]


@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_m[i, j] = 0.0
        grid_v_in[i, j] = [0.0,0.0]
        grid_c[i, j] = 0.0

        grid_m.grad[i, j] = 0.0
        grid_v_in.grad[i, j] = [0.0,0.0]
        grid_v_out.grad[i, j] = [0.0,0.0]
        grid_c.grad[i, j] = 0.0

@ti.kernel
def clear_particle_grad(): 
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = zero_vec()
        v.grad[f, i] = zero_vec()
        Jp.grad[f, i] = 0.0
        C.grad[f, i] = zero_matrix()
        F.grad[f, i] = zero_matrix()

# --------------- Physical Simulatiom ---------------

@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        new_Jp = Jp[f, p]
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)

        w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)] # quadratic kernels
        new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p] # deformation gradient update
        h = max(0.1, min(5, ti.exp(10 * (1.0 - new_Jp))))

        if particle_type[p] == 1: # jelly, make it softer
            h = 0.3

        mu, la = mu_0 * h, lambda_0 * h
        if particle_type[p] == 0: # liquid
            mu = 0.0

        U, sig, V = ti.svd(new_F)

        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if particle_type[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            new_Jp *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if particle_type[p] == 0:  # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.diag(dim=dim, val=1) * ti.sqrt(J)
        elif particle_type[p] == 2:
            new_F = U @ sig @ ti.transposed(V) # Reconstruct elastic deformation gradient after plasticity
        
        F[f + 1, p] = new_F
        Jp[f + 1, p] = new_Jp

        stress = 2 * mu * (new_F - U @ ti.transposed(V)) @ ti.transposed(new_F) + ti.Matrix.diag(dim=dim, val=1) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[f, p]

        # loop over 3x3 node neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * dx
                weight = w[i](0) * w[j](1)
                ti.atomic_add(grid_v_in[base + offset], weight * (p_mass * v[f, p] + affine @ dpos))
                ti.atomic_add(grid_m[base + offset], weight * p_mass)


bound = 3

@ti.kernel
def grid_op():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            v_out = inv_m * grid_v_in[i, j] # momentum to velocity
            v_out[1] -= dt * gravity # gravity

            # center sticky circle
            dist = ti.Vector([i * dx - 0.5, j * dx - 0.5])
            if dist.norm_sqr() < 0.005:
                dist = ti.normalized(dist)
                v_out -= dist * ti.dot(v_out, dist)

            # boundary conditions
            if i < bound and v_out[0] < 0:
                v_out[0] = 0
            if i > n_grid - bound and v_out[0] > 0:
                v_out[0] = 0
            if j < bound and v_out[1] < 0:
                v_out[1] = 0
            if j > n_grid - bound and v_out[1] > 0:
                v_out[1] = 0

            grid_v_out[i,j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        # loop over 3x3 neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
                g_v = grid_v_out[base(0) + i, base(1) + j]
                weight = w[i](0) * w[j](1)
                new_v += weight * g_v
                new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx

        v[f+1, p] = new_v
        x[f+1, p] = x[f, p] + dt * v[f+1, p] # advection
        C[f+1, p] = new_C

@ti.kernel
def compute_stuck(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        p_stuck = False
        # loop over 3x3 neighborhood
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
                # if particle is jelly, update grid_c to account for presence of particle
                if particle_type[p] == 1 and grid_c[base + offset] == 0:
                    grid_c[base + offset] = 1
                if particle_type[p] == 0 and grid_c[base + offset] > 0 and not p_stuck:
                    p_stuck = True
        if p_stuck: stuck[f] += 1

mesh = lambda i, j: i * n_particle_y + j
fluidmesh = lambda i, j: i * n_fluid_particles_y + j


# --------------- Initialize Scene ---------------
class Scene():
    def __init__(self):
        self.n_particles = 0
        self.n_container_particles = 0
        self.x = []
        self.v = []
        self.actuator_id = []
        self.particle_type = []
        self.num_actuators = 0 

    def new_actuator(self):
        self.num_actuators += 1
        global n_actuators
        n_actuators = self.num_actuators
        return self.num_actuators - 1

    def clear(self):
        '''
        removes all objects' particle positions and velocities
        for resetting of positions
        '''
        self.x = []
        self.v = []
        self.particle_type = []
        self.n_particles = 0


    def add_block(self, x, y, w, h, v_x, v_y, actuation, ptype=0):
        # (x, y) define left corner!
        # default particle type is liquid
        mesh = lambda i, j: i * h + j
        ids = []
        xm, vm = [], []
        for i in range(w):
            for j in range(h):
                t = mesh(i, j)
                xm.append([x + i * dx * 0.5, y + j * dx * 0.5])
                vm.append([v_x, v_y])

                ids.append(t)
                self.particle_type.append(ptype)
                self.n_particles += 1
        sorted_ids = np.argsort(np.array(ids))
        xm = np.array(xm)[sorted_ids]
        vm = np.array(vm)[sorted_ids]
        self.x += xm.tolist()
        self.v += vm.tolist()

    def add_container(self, x, y, v_x, v_y, w, h, d, actuation, ptype=1):
        # (x, y) define left corner!
        # w, h define width and height in particles (respectively)
        # d defines thickness of walls
        # If d==h, the 'container' has no walls.
        # Note: particle type 1 is reserved for the container object!

        bottom = lambda i, j: i * d + j
        wall = lambda i, j: i * (h-d) + j

        # build bottom
        ids, xm, vm = [], [], []
        for i in range(w):
            for j in range(d):
                t = bottom(i, j)
                xm.append([x + i * dx * 0.5, y + j * dx * 0.5])
                vm.append([v_x, v_y])

                ids.append(t)
                self.particle_type.append(ptype)
                self.n_container_particles += 1
        
        if h > d:
            # build left wall
            for i in range(d):
                for j in range(d, h):
                    t = wall(i, j) + w*d
                    xm.append([x + i * dx * 0.5, y + j * dx * 0.5])
                    vm.append([v_x, v_y])

                    ids.append(t)
                    self.particle_type.append(ptype)
                    self.n_container_particles += 1

             # build right wall
            for i in range(w-d, w):
                for j in range(d, h):
                    t = wall(i, j) + w*d + d*(h-d)
                    xm.append([x + i * dx * 0.5, y + j * dx * 0.5])
                    vm.append([v_x, v_y])

                    ids.append(t)
                    self.particle_type.append(ptype)
                    self.n_container_particles += 1

        sorted_ids = np.argsort(np.array(ids))
        xm = np.array(xm)[sorted_ids]
        vm = np.array(vm)[sorted_ids]
        self.x += xm.tolist()
        self.v += vm.tolist()
        self.n_particles += self.n_container_particles


    #  ======= Once all blocks are added to scene, update global class vars
    def finalize(self):
        global n_particles, n_container_particles
        n_particles = self.n_particles
        n_container_particles = self.n_container_particles
        print('n_particles', n_particles)
        print('n_container_particles', n_container_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act
    #  ========

def initialize_env(scene):
    #NOTE: ALWAYS ADD CONTAINER FIRST 
    scene.add_container(0.5, floor, 0, 0, 60, 40, 10, 1, ptype=1)
    scene.add_block(0.5, 0.7, n_fluid_particles_x*4, n_fluid_particles_y, 0, -1, 0, ptype=0)     

# --------------- Visualization ---------------
def make_fig(particle_pos, ids, file_path, stuck):
    fig, ax = plt.subplots()
    ax.scatter(particle_pos[ids==0, 0, :], particle_pos[ids==0, 1, :], s=2.5, c='m', alpha=0.3, edgecolors='k', linewidths=0.1)
    ax.scatter(particle_pos[ids==1, 0, :], particle_pos[ids==1, 1, :], s=2, c='g', alpha=0.4, edgecolors='k', linewidths=0.1)
    ax.scatter(particle_pos[ids==2, 0, :], particle_pos[ids==2, 1, :], s=2, c='b', alpha=0.4, edgecolors='k', linewidths=0.1)
    circle = plt.Circle((0.5, 0.5), radius=0.07, color='k', alpha=0.4)
    ax.add_artist(circle)

    l1 = mlines.Line2D([0,1.0], [0.0125,0.0125], color='k', alpha=0.4)
    l2 = mlines.Line2D([1.0,1.0], [0.0125,1.0], color='k', alpha=0.4)
    plt.text(0.7, 0.95, 'stuck = '+str(stuck))
    ax.add_line(l1)
    ax.add_line(l2)
    ax.set(xlim=(0.0, 1.0), ylim = (0, 1))
    ax.set_aspect('equal', 'box')
    

    plt.savefig(file_path)
    plt.close()

# --------------- Training Functions ---------------

def clear_physical_params(scene):
    for t in range(scene.n_particles):
        x[0, t] = scene.x[t]
        v[0, t] = scene.v[t]
        F[0, t] = [[1, 0], [0, 1]]
        Jp[0, t] = 1
        C[0, t] = [[0.0, 0.0], [0.0, 0.0]]
        particle_type[t] = scene.particle_type[t]

@ti.kernel
def compute_loss(t: ti.i32):
    # for now, compute loss simply as number of particles stuck to container object
    # at final timestep
    val = stuck[t]
    loss[None] = -val

@ti.kernel
def reset():
    for p in range(n_container_particles):
        v[0, p] = [init_v[None], 0]


def forward(output, visualize=True, viz_int=50):
    if visualize and not (output is None):
        os.makedirs(output, exist_ok=True)
        filenames = []

    # simulation
    reset()
    for s in range(max_steps):
        clear_grid()
        p2g(s)
        grid_op()
        g2p(s)
        compute_stuck(s)

        if visualize and not (output is None):
            if s % viz_int == 0:
                file_path = os.path.join(output, 'stp{:04d}'.format(int(s/viz_int)))
                make_fig(x.to_numpy()[s,:,:,:], particle_type.to_numpy(), file_path, stuck.to_numpy()[s])
                filenames.append(file_path+'.png')
            if s == max_steps-1:
                gif_path = os.path.join(output, 'mov.gif')
                with imageio.get_writer(gif_path, mode='I') as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
        
        print(' Progress: %8.2f%%.' % (s*100.0/max_steps), end='\r')
    
    compute_loss(max_steps-1)
    return loss[None]

def backward():
    clear_particle_grad()

    compute_loss.grad(max_steps-1)
    for s in reversed(range(max_steps)):
        pdb.set_trace()
        clear_grid()
        p2g(s)
        grid_op()

        compute_stuck.grad(s)
        g2p.grad(s)
        grid_op.grad()
        p2g.grad(s)


def main():
    # set initial values of x component of velocity for container
    init_v[None] = 0.1

    # INITIALIZE SCENE
    scene = Scene()
    initialize_env(scene)
    scene.finalize() # wrap up intialization by reupdating global scene parm
    clear_physical_params(scene) # clear physical parameters for new simulation

    losses = []
    init_v.grad[None]=1
    for n in range(num_iters):
        # output defines the name of the directory where files
        stuck.fill(0)
        ti.clear_all_gradients()
        if n % 1 == 0:
            output = 'obst_figs/iter{:04d}'.format(n) 
        else:
            output = None

        
        with ti.Tape(loss):
            forward(output, visualize=False)

        print('Iter =', n, 'Loss =', loss[None], '                  ')
        
        init_v[None] -= learning_rate * init_v.grad[None]
        print('init_v =', init_v[None], '                  ')
        print('init_v.grad =', init_v.grad[None], '                  ')


if __name__ == '__main__':
    main()
