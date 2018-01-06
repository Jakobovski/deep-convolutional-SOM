from __future__ import division
import math
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from alt_backprop.models.subfield import Subfield
import alt_backprop.config as cfg
matplotlib.use('TkAgg')


class ReceptiveField2D(object):
    # The dimensions of pixels in the input data that this RF accepts
    dimensionality = 2

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
        self.input_side = input_shape[0]


        # Setup the dict to hold the subfields
        self.subfields = {}

        # Initialize the subfield
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.subfields[position] = Subfield(position, self.input_shape, self)

    def accept_input(self, signal, learn):
        for pos, subfield in self.subfields.iteritems():
            subfield.get_excitation(signal)

        max_subfield = max(self.subfields.values(), key=lambda csubfield: csubfield.excitation)
        if learn:
            self._run_learning(max_subfield, signal)

        return max_subfield

    def get_raw_representation_shape(self):
        if self.parent_field:
            d = self.input_shape[0] * self.parent_field.get_raw_representation_shape()[0]
        else:
            d = self.input_shape[0]
        return (d, d)

    def get_image_representation(self, weight):
        if self.parent_field:
            subfield = self.subfields[(int(round(weight[0])), int(round(weight[1])))]
            full_image = np.zeros(self.get_raw_representation_shape())
            for x_idx, col in enumerate(subfield.weights):
                for y_idx, w in enumerate(col):
                    img = self.parent_field.get_image_representation(w)
                    start_x = x_idx * img.shape[0]
                    end_x = (x_idx + 1) * img.shape[0]
                    start_y = y_idx * img.shape[0]
                    end_y = (y_idx + 1) * img.shape[0]
                    full_image[start_x:end_x, start_y:end_y] = img
            return full_image
        else:
            # Find the subfield at pos=weight
            assert len(weight) == 2
            # TODO: may want to average the nearest neighbors instead of rounding
            subfield = self.subfields[int(round(weight[0])), int(round(weight[1]))]
            return subfield.weights

    def _run_learning(self, max_subfield, signal):
        # Get the subfield with the greatest excitation
        max_subfield.move_towards(signal, cfg.LEARNING_RATE)  # TODO. We might have double learning here.. Thats probably OK

        # Find the neighbors of the recently fired subfields and make them learn a bit
        for neighbor in max_subfield.get_neighbor_cords(0, cfg.POS_NEIGHBOR_MAX_PERCENT):
            dx = abs(max_subfield.position[0] - neighbor[0])
            dy = abs(max_subfield.position[1] - neighbor[1])
            learning_rate = cfg.NEIGHBOR_LEARNING_RATE / (1 + math.sqrt(dy**2 + dx**2))
            self.subfields[neighbor].move_towards(signal, learning_rate=learning_rate)

    def visualize(self):
        """Displays an image of a receptive field"""
        if self.dimensionality == 1:
            dim_input = self.input_shape[0] # this is the number of pixels per side for each sub-image
            d = self.side_len * self.input_shape[0]  # this is total image pixels per side
            large_image = np.zeros((d, d))

            for pos, csubfield in self.subfields.iteritems():
                img = csubfield.weights
                offset = [pos[0] * dim_input, pos[1] * dim_input]
                large_image[offset[0]:offset[0] + dim_input, offset[1]:offset[1] + dim_input] = img

            fig = plt.figure()
            plt.imshow(large_image, cmap='Greys_r', interpolation='none')
            plt.show(block=False)

            # Make the window on top
            if matplotlib.get_backend() == 'TkAgg':
                fig.canvas.manager.window.attributes('-topmost', 1)
            else:
                fig.window.raise_()

        else:
            # What we want to do here is draw the image that excites each of this receptive fields subfields the most
            # So the dimensions of this image are as follows
            subfield_len = self.get_raw_representation_shape()[0]  # 8
            d = self.side_len * subfield_len
            large_image = np.zeros((d, d))

            for pos, subfield in self.subfields.iteritems():
                sub_field_image = np.zeros((subfield_len, subfield_len))
                for x_idx in range(self.input_shape[0]):
                    for y_idx in range(self.input_shape[0]):
                        weight = subfield.weights[x_idx][y_idx]
                        assert len(weight) == 2
                        # find the image representation in the parent receptive filed
                        img = self.parent_field.get_image_representation(weight)
                        start_x = x_idx * img.shape[0]
                        end_x = (x_idx+1) * img.shape[0]
                        start_y = y_idx * img.shape[0]
                        end_y = (y_idx+1) * img.shape[0]
                        sub_field_image[start_x:end_x, start_y:end_y] = img

                start_x = pos[0] * subfield_len
                start_y = pos[1] * subfield_len
                end_x = (pos[0]+1) * subfield_len
                end_y = (pos[1]+1) * subfield_len
                large_image[start_x:end_x, start_y:end_y] = sub_field_image

            fig = plt.figure( figsize=(25, 25))
            plt.imshow(large_image, cmap='Greys_r', interpolation='none')
            plt.show(block=False)

            # # Make the window on top
            # if matplotlib.get_backend() == 'TkAgg':
            #     fig.canvas.manager.window.attributes('-topmost', 1)
            # else:
            #     fig.window.raise_()
