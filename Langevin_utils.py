import numpy as np
import matplotlib.pyplot as plt
# this is step A
#this function returns the energy and force on a particle from a harmonic potential
def harmonic_oscillator_energy_force(x,k=1,x0=0):
    #calculate the energy on force on the right hand side of the equal signs
    energy = 0.5*k*(x-x0)**2
    force = -k*(x-x0)
    return energy, force

#this function will plot the energy and force
#it is very general since it uses a special python trick of taking arbitrary named arguments (**kwargs)
#and passes them on to a specified input function
def plot_energy_force(function, xmin=-3,xmax=3,spacing=0.1,**kwargs):
    x_points = np.arange(xmin,xmax+spacing,spacing)
    energies, forces = function(x_points,**kwargs)
    label = 'U(x)'
    for arg in kwargs:
        label=label+', %s=%s'%(arg,str(kwargs[arg]))
    p = plt.plot(x_points,energies,label=label)
    plt.plot(x_points,forces,label='',color=p[0].get_color(),linestyle='--')
    plt.legend(loc=0)
def position_update(x, v, dt):
    x_new = x + v * dt / 2.
    return x_new


# this is step B
def velocity_update(v, F, dt):
    v_new = v + F * dt / 2.
    return v_new


def random_velocity_update(v, gamma, kBT, dt):
    R = np.random.normal()
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1 - c1 * c1) * np.sqrt(kBT)
    v_new = c1 * v + R * c2
    return v_new


def baoab(potential, max_time, dt, gamma, kBT, initial_position, initial_velocity,
          save_frequency=3, **kwargs):
    x = initial_position
    v = initial_velocity
    t = 0
    step_number = 0
    positions = []
    velocities = []
    total_energies = []
    save_times = []

    while (t < max_time):

        # B
        potential_energy, force = potential(x, **kwargs)
        v = velocity_update(v, force, dt)

        # A
        x = position_update(x, v, dt)

        # O
        v = random_velocity_update(v, gamma, kBT, dt)

        # A
        x = position_update(x, v, dt)

        # B
        potential_energy, force = potential(x, **kwargs)
        v = velocity_update(v, force, dt)

        if step_number % save_frequency == 0 and step_number > 0:
            e_total = .5 * v * v + potential_energy

            positions.append(x)
            velocities.append(v)
            total_energies.append(e_total)
            save_times.append(t)

        t = t + dt
        step_number = step_number + 1

    return save_times, positions, velocities, total_energies

def generate_spikes(xs,vs,number_of_observations, min_sigma,max_hist_dependency,max_firing_rate):

        max_sigma = (xs.max() - xs.min()) / 20

        bfr = np.random.uniform(0, high=max_firing_rate, size=number_of_observations)

        mu_x = np.random.uniform(low=xs.min(), high=xs.max(),size=number_of_observations)
        sigma_x = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)

        mu_v = np.random.uniform(low=vs.min(), high=vs.max(), size=number_of_observations)
        sigma_v = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)


        Lambdas1 = np.exp(bfr - (xs - mu_x) ** 2 / (2 * sigma_x ** 2) )
        Lambdas2 = np.exp(bfr - (vs - mu_v) ** 2 / (2 * sigma_v ** 2))
        Spikes = (np.random.poisson(Lambdas1+Lambdas2))
        Spikes[Spikes > 1] = 1

        Lambdas2 = np.zeros_like(Lambdas1)


        sigma_eff = np.random.uniform(low=min_sigma, high=max_sigma, size=number_of_observations)
        rt = np.random.uniform(0, 0, size=number_of_observations)
        for ii in range(number_of_observations):
            qq=np.random.randint(0,number_of_observations)
            si=np.where(Spikes[:,qq] >0)[0]

            Lambdas2[:,ii] = (1-np.exp(-
                                    (np.arange(Spikes.shape[0]).reshape([-1,1]) - si.reshape([1,-1]) -rt[ii]) ** 2 / (2 * sigma_eff[ii] ** 2) )).mean(axis=-1).squeeze()

        Spikes = (np.random.poisson(Lambdas1*Lambdas2))
        Spikes[Spikes > 1] = 1
        f, axes = plt.subplots(2, 1, sharex=True, sharey=False)
        axes[0].plot(mu_x, 'o', label='mu_x')
        axes[0].set_title('mu_x')
        axes[1].plot(mu_v, 'o', label='mu_v')
        axes[1].set_title('mu_v')
        return Spikes

