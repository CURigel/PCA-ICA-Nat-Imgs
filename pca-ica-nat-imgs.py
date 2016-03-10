__author__ = 'Devin'

import os
import re
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from random import randint
from sklearn.decomposition import FastICA, PCA

DEBUG_MODE = True
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Borrowed from assignment 5
# dirname: directory to locate the image files
# ('./Opencountry' by default.)
def load_images(dirname='./Opencountry'):
    files = [dirname + '/' + f for f in os.listdir(dirname) \
                               if re.search('.*' + '('+ '|'.join(IMG_EXTENSIONS) + ')', f)]
    files.sort()

    images = [misc.imread(f, True) for f in files] # Second (optional) argument in imread, 'flatten=True',
                                                   # converts the read image to grayscale.
                                                   # Remove if you'd like to retain image in RGB.
    return images


# Borrowed from assignment 5
# patchSize: a 2-tuple of (height, width) of the desired patches
# sampleCount: # of random samples to collect
def get_image_sample(images, patchSize, sampleCount):
    samples = []
    for i in range(sampleCount):
        randidx = randint(0, len(images)-1) # Select an image randomly
        img_height, img_width = images[randidx].shape

        while patchSize[0] > img_height or patchSize[1] > img_width:
            print 'Patch size larger than image: Retrying...'
            randidx = randint(0, len(images)-1) # Reselect an image
            img_height, img_width = images[randidx].shape

        # (y,x) coordinates of top left corner of the patch
        randPatchStart = (randint(0, img_height-patchSize[0]), randint(0, img_width-patchSize[1]))
        samples.append(images[randidx][randPatchStart[0]:randPatchStart[0]+patchSize[0], \
                                       randPatchStart[1]:randPatchStart[1]+patchSize[1]])

    return samples

def generate_rand_component_model(num_components, num_points, **kwargs):
    comp_type = kwargs['comp_type'] if 'comp_type' in kwargs else 'repeat_one'
    noise = kwargs['noise'] if 'noise' in kwargs else False
    noise_scale = kwargs['noise_scale'] if 'noise' in kwargs else 0.1
    repeat_sequence_length = kwargs['repeat_sequence_length'] if 'repeat_sequence_length' in kwargs else 100
    component_means = np.asarray(kwargs['component_means']) if 'component_means' in kwargs else np.repeat(0.5, num_components)
    component_scales = np.asarray(kwargs['component_scales']) if 'component_scales' in kwargs else np.repeat(1, num_components)
    take_sin = kwargs['take_sin'] if 'take_sin' in kwargs else False

    assert(component_means.shape[0] == num_components)
    assert(component_scales.shape[0] == num_components)

    if comp_type == 'all_random':
        components = np.random.rand(num_points, num_components) * np.tile((2 * component_means), (num_points, 1))
    elif comp_type == 'repeat_one':
        components = np.tile(np.random.rand(num_components) * (2 * component_means), (num_points, 1))
    elif comp_type == 'repeat_sequence':
        repeating_sections = np.random.rand(repeat_sequence_length, num_components) * \
                             np.tile((2 * component_means), (repeat_sequence_length, 1))
        components = np.tile(repeating_sections, ((num_points / repeat_sequence_length)+1, 1))[0:num_points, :]
    elif comp_type == 'laplace':
        component_list = []
        for i in np.arange(num_components):
            laplace = np.random.laplace(component_means[i], component_scales[i], num_points)
            component_list += [laplace]
        components = np.asarray(component_list).T
    elif comp_type == 'combined':
        assert(num_components == 3)
        component_list = []
        component_list += [np.random.laplace(component_means[0], component_scales[0], num_points)]
        component_list += [(np.random.rand(1) * (2 * component_means[1])).repeat(num_points)]
        repeating = np.random.rand(repeat_sequence_length) * \
                    np.repeat((2 * component_means[2]), repeat_sequence_length)
        component_list += [np.repeat(repeating, (num_points / repeat_sequence_length)+1)[0:num_points]]
        components = np.asarray(component_list).T
    else:
        raise Exception('unknown type')

    if take_sin:
        components += np.tile(np.c_[np.arange(num_points)] / 100.0, (1, num_components))
        components = np.sin(components)

    mixing_mat = np.random.rand(num_components, num_components) + 1
    observations = components.dot(mixing_mat)

    if noise:
        noise_values = noise_scale * np.random.normal(size=observations.shape)
        observations += noise_values

    if DEBUG_MODE:
        assert((observations.shape == np.asarray([num_points, num_components])).all())
        obs_should_be = (mixing_mat[:, 0] * components[0, :]).sum() + (noise_values[0, 0] if noise else 0)
        assert(np.allclose(obs_should_be, observations[0, 0]))

    return components, mixing_mat, observations


def example_model():
    # example model data from
    # http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations
    return S, A, X


def plot_results(observations, true_components, ica_components, pca_components):
    plt.figure()

    models = [observations, true_components, ica_components, pca_components]
    names = ['Observations',
             'True Sources',
             'ICA recovered sources',
             'PCA recovered sources']
    colors = ['red', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def run_ica(num_components, data):
    ica = FastICA(n_components=num_components)
    ica_component_estimate = ica.fit_transform(data)
    ica_mixing_estimate = ica.mixing_
    return ica_component_estimate, ica_mixing_estimate


def run_pca(num_components, data):
    pca = PCA(n_components=num_components)
    pca_estimate = pca.fit_transform(data)
    return pca_estimate


def main():
    rand_components, rand_mixing, rand_data = \
        generate_rand_component_model(3, 2000, comp_type='combined', component_means=[1., 2., 3.],
                                      component_scales=[0.5, 0.5, 0.5], take_sin=True)

    ica_component, ica_mixing = run_ica(3, rand_data)
    pca_component = run_pca(3, rand_data)

    plot_results(rand_data, rand_components, ica_component, pca_component)


if __name__ == "__main__":
    main()
