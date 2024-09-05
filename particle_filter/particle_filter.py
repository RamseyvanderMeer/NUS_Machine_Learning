import numpy as np
import scipy.stats
from sklearn.preprocessing import normalize
import math
import random

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits
    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound
    # (jump from -pi to pi). Therefore, we generate unit vectors from the
    # angles and calculate the angle of their average

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans

    # generate new particle set after motion update
    new_particles = []

    for particle in particles:
        new_particle = dict()
        #sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

        #calculate new particle pose
        new_particle['x'] = particle['x'] + \
            noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        new_particle['y'] = particle['y'] + \
            noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        new_particle['theta'] = particle['theta'] + \
            noisy_delta_rot1 + noisy_delta_rot2
        new_particles.append(new_particle)
    return new_particles

def importance_weight(sigma_r, x, u):
    # print((1/np.sqrt(2 * math.pi * sigma_r**2)) * math.e ** -(x - u)**2 / (2 * u**2))
    return (1/np.sqrt(2 * math.pi * sigma_r**2)) * math.e ** -(x - u)**2 / (2 * u**2)


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id'] # [1, 2, 3, 4]
    ranges = sensor_data['range'] # [3.9421015658823655, 5.969574284580203, 6.189846553236058, 3.461872331368626]

    weights = []

    '''your code here'''
    '''***        ***'''

    # calculate the weights
    for particle in particles:
        norm = []
        for landmark in landmarks:

            # calculate distance to landmark
            x_wheight = particle['x'] - landmarks[landmark][0]
            # if landmarks[landmark][0] == 0 or particle['x'] == 0:
            #     print(landmark, landmarks[landmark], landmarks[landmark][0], particle['x'])
            y_wheight = particle['y'] - landmarks[landmark][1]
            dist  = np.sqrt(x_wheight**2 + y_wheight**2)
            norm.append(importance_weight(sigma_r, dist, ranges[landmark-1]))
            # print(dist)
        # weights.append(np.sum(norm))
        weights.append(np.prod(norm) ** 25)

    # weights = [weight**50 for weight in weights]

    #normalize weights
    normalizer = sum(weights)
    weights = [weight / normalizer for weight in weights]
    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    '''your code here'''
    '''***        ***'''

    # print(sum(weights))
    choice_arr = []
    sum = 0
    for val in weights:
        sum += val
        choice_arr.append(sum)

    # for idx in range(len(particles)):
    #     # add particle to new_particles if it is selected
    #     val = random.random()
    #     for i, choice in enumerate(choice_arr):
    #         if val < choice:
    #             new_particles.append(particles[i])
    #             break

    # return new_particles

    for idx in range(len(particles) // 4):
        # add particle to new_particles if it is selected
        val = random.random()
        for i, choice in enumerate(choice_arr):
            if val < choice:
                new_particles.append(particles[i])
                break

        new_val = np.abs(1-val)
        for i, choice in enumerate(choice_arr):
            if new_val < choice:
                new_particles.append(particles[i])
                break

        new_val = np.abs(1-val/2)
        for i, choice in enumerate(choice_arr):
            if new_val < choice:
                new_particles.append(particles[i])
                break

        new_val = np.abs(1-val*3/4)
        for i, choice in enumerate(choice_arr):
            if new_val < choice:
                new_particles.append(particles[i])
                break



    return new_particles

