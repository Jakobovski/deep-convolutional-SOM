from __future__ import division
import numpy as np


class Subfield(object):

    def __init__(self, position, shape, receptive_field):
        self.position = position
        self.shape = shape
        self.weights = np.random.uniform(0, 5, shape)
        self.receptive_field = receptive_field

        # A dictionary to cache results form get_neighbor_cords()
        self._cache = {}
        self.excitation = 0

    def get_excitation(self, image):
        self.excitation = (1 / np.linalg.norm(self.weights - image))
        return self.excitation

    def move_towards(self, image, learning_rate):
        """ Makes the neurons weights move toward the passed image"""
        if image.shape != self.weights.shape:
            raise Exception("Image shape does not match weights shape", image.shape, self.weights.shape)
        self.weights += (image - self.weights) * learning_rate

    def get_neighbor_cords(self, nmin, nmax_percent):
        """
        nmin: The neighbors that are at least 1 distance from the position.
        nmax: the max distance to get neighbors, (not inclusive.).
        returns the coordinates of neighbors that are at least `min` distance, but less than `max` distance.
        """
        nmax = int(nmax_percent * self.receptive_field.side_len)
        neighbors = self._cache.get((nmin, nmax), None)

        if not neighbors:
            # print 'Not using cache. (First time neuron fired)'
            side_length = self.receptive_field.side_len
            i, j = self.position
            neighbors = []

            x_range = [i - nmax, i + nmax]
            y_range = [j - nmax, j + nmax]

            for x in range(x_range[0] + 1, x_range[1]):
                for y in range(y_range[0] + 1, y_range[1]):
                    pos = (x, y)
                    dx = abs(x - i)
                    dy = abs(y - j)

                    if dx < nmin and dy < nmin:
                        continue
                    elif pos[0] < 0 or pos[0] > side_length - 1:
                        continue
                    elif pos[1] < 0 or pos[1] > side_length - 1:
                        continue
                    else:
                        neighbors.append(pos)
            self._cache[(nmin, nmax)] = neighbors

        return neighbors
