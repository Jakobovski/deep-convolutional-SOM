from __future__ import division
import math
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from alt_backprop.models.subfield import Subfield
import alt_backprop.config as cfg
matplotlib.use('TkAgg')


class ReceptiveField1D(object):
    # The dimensions of pixels in the input data that this RF accepts
    dimensionality = 1

    def __init__(self, num_subfields, input_shape, parent_field):
        """A receptive field that excepts 1D values for each input pixel. This is best used for grayscale images, color
            images will use a 3D receptive field (rgb)

        This represents a small region of input space. A RF consists of many subfields, these subfields
        compete with each other to learn representations.

        It can be thought of as a group of subfields.
        """

        self.num_subfields = num_subfields
        self.parent_field = parent_field
        # The subfields need to be a square
        self.side_len = int(math.sqrt(num_subfields))
        self.shape = (self.side_len, self.side_len)
        assert math.sqrt(self.num_subfields) % 1 == 0

        # This is the shape of the image that is input into the receptive field
        self.input_shape = input_shape

        # Setup the dict to hold the subfields
        self.subfields = {}
        self.learning_enabled = True

        # Initialize the subfield
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.subfields[position] = Subfield(position, self.input_shape, self)

    def accept_input(self, input_image, learn):
        for pos, conv_subfield in self.subfields.iteritems():
            conv_subfield.get_excitation(input_image)

        max_subfield = max(self.subfields.values(), key=lambda csubfield: csubfield.excitation)
        if learn and self.learning_enabled:
            self._run_learning(max_subfield, input_image)

        return max_subfield

    def get_raw_representation_shape(self):
        if self.parent_field:
            d = self.input_shape[0] * self.parent_field.get_raw_representation_shape()[0]
        else:
            d = self.input_shape[0]
        return (d, d)

    def get_image_representation(self, weight):
        if self.parent_field:
            raise NotImplementedError
        else:
            # Find the subfield at pos=weight
            assert len(weight) == 2
            # TODO: may want to average the nearest neighbors instead of rounding
            subfield = self.subfields[(int(round(weight[0])), int(round(weight[1])))]
            return subfield.weights

    def _run_learning(self, max_subfield, input_image):
        # Get the subfield with the greatest excitation
        max_subfield.move_towards(input_image, cfg.LEARNING_RATE)  # TODO. We might have double learning here.. Thats probably OK

        # Find the neighbors of the recently fired subfields and make them learn a bit
        for neighbor in max_subfield.get_neighbor_cords(0, cfg.POS_NEIGHBOR_MAX_PERCENT):
            dx = abs(max_subfield.position[0] - neighbor[0])
            dy = abs(max_subfield.position[1] - neighbor[1])
            learning_rate = cfg.NEIGHBOR_LEARNING_RATE / (1 + math.sqrt(dy**2 + dx**2))
            self.subfields[neighbor].move_towards(input_image, learning_rate=learning_rate)

    def visualize(self):
        """Displays an image of a receptive field"""
        dsubfield = self.input_shape[0]
        d = self.side_len * self.input_shape[0]
        large_image = np.zeros((d, d))

        for pos, csubfield in self.subfields.iteritems():
            img = csubfield.weights
            offset = [pos[0] * dsubfield, pos[1] * dsubfield]
            large_image[offset[0]:offset[0] + dsubfield, offset[1]:offset[1] + dsubfield] = img

        fig = plt.figure()
        plt.imshow(large_image, cmap='Greys_r', interpolation='none')
        plt.show(block=False)
        #
        # # Make the window on top
        # if matplotlib.get_backend() == 'TkAgg':
        #     fig.canvas.manager.window.attributes('-topmost', 1)
        # else:
        #     fig.window.raise_()
