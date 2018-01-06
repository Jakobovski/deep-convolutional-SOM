from __future__ import division
import math
import numpy as np
from skimage.util.shape import view_as_blocks

from alt_backprop.utils import utils, patch_util
import alt_backprop.config as config


class Layer(object):
    def __init__(self, receptive_field, input_dims, name=None):
        """ A layer contains one receptive field that is slid over the input image. This is similar to a filter in a
        convolution neural network.
        """
        assert input_dims[0] == input_dims[1]
        self.input_dims = input_dims
        self.receptive_field = receptive_field

        self.output_layer = None
        self.input_layer = None
        self.name = name

    def set_learning_rate(self, rate):
        self.receptive_field.set_learning_rate(rate)

    def accept_input(self, input_image, learn=True):
        """ Takes an input and learns it, passes it to the next layer"""
        if config.MODE == 'DEBUG':
            if np.shape(input_image)[0] != np.shape(input_image)[1]:
                raise Exception('Input input_image must be square.')

            if self.receptive_field.dimensionality == 1:
                if np.shape(input_image) != self.input_dims:
                    raise Exception('Input input_image is of the wrong shape.')
            else:
                if np.shape(input_image) != (
                self.input_dims[0], self.input_dims[0], self.receptive_field.dimensionality):
                    raise Exception('Input input_image is of the wrong shape.')

            # Take the first pixel of the input image. If it's OK then all pixels are probably OK.
            if hasattr(input_image[0][0], '__iter__'):
                if len(input_image[0][0]) != self.receptive_field.dimensionality:
                    raise Exception('Input input_image is of the wrong dimensionality. Received {}, Expected: {}'
                                    .format(len(input_image[0][0]), self.receptive_field.dimensionality))
            else:
                if self.receptive_field.dimensionality != 1:
                    raise Exception('Input input_image is of the wrong dimensionality. Received {}, Expected: {}'
                                    .format(1, self.receptive_field.dimensionality))

        # Adds padding to the image
        padding_needed = int(len(input_image[0]) % self.receptive_field.input_shape[0])
        padding_before = int(math.ceil(padding_needed / 2.0))
        padding_after = int(padding_needed / 2.0)
        assert padding_needed == (padding_before + padding_after)
        npad = [(padding_before, padding_after), (padding_before, padding_after)]
        if hasattr(input_image[0][0], '__iter__'):  # TODO remove once all images are 3d
            npad.append((0, 0))

        pad_value = (0,0)
        if len(input_image.shape) > 2:
            if input_image.shape[2] != len(pad_value):
                raise Exception('Padding with the wrong value')

        padded_image = np.pad(input_image, pad_width=npad, mode='constant', constant_values=pad_value)

        # Split the input up into patches
        patches = patch_util.extract_patches(padded_image, self.receptive_field.input_shape)

        # Send those patches to the receptive field, and get the neuron that was most strongly excited
        # for each patch
        excited_subfields = []
        for patch in patches:
            sf = self.receptive_field.accept_input(patch, learn=learn)
            excited_subfields.append(sf)

        if self.output_layer:
            # Now we know the neuron that was excited for each receptive field in the input image
            # Create a new 'image' where each pixel contains a 2d position value, this value corresponds to the
            # position of the excited neuron.
            self.output_layer.accept_input(self.prep_for_output_layer(excited_subfields), learn=learn)

        return excited_subfields

    def prep_for_output_layer(self, excited_subfields):
        positions = [sf.position for sf in excited_subfields]
        side_len = int(math.sqrt(len(positions)))
        return np.reshape(positions, (side_len, side_len, self.output_layer.receptive_field.dimensionality))

    def visualize(self):
        self.receptive_field.visualize()

        if self.output_layer:
            self.output_layer.visualize()

    def set_output_layer(self, layer):
        self.output_layer = layer

    def set_input_layer(self, layer):
        self.input_layer = layer

    def recreate_input(self, image):
        excited_subfields = self.accept_input(image, learn=False)
        if self.output_layer is None:
        # if True:
            images = []
            for sf in excited_subfields:
                images.append(self.receptive_field.get_image_representation(sf.position))

            return utils.stitch_images(images)

        if self.output_layer is not None:
            preped = self.prep_for_output_layer(excited_subfields)
            return self.output_layer.recreate_input(preped)
