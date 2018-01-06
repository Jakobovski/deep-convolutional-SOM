import os
import numpy as np
# import scipy.ndimage as ndimage



def extract_patches(image, desired_shape):
    """Accepts an image and extracts patches from top-left to bottom-right iterating columns then rows. It does not
    patch along depth(if exists).
    """
    if image.shape[0] != image.shape[1]:
        raise Exception("Image is not square along first 2 dimensions", image.shape)

    if len(desired_shape) != len(image.shape):
        raise Exception("Length of image shape and desired shape do not match", image.shape, desired_shape)

    if len(desired_shape) == 3:
        if desired_shape[2] != image.shape[2]:
            raise Exception("Image depth and desired depth do not match", image.shape, desired_shape)

    # Note: only the first check is necessary, (because of the checks above) but we do both just to be safe.
    if image.shape[0] % desired_shape[0] != 0 or image.shape[1] % desired_shape[1] != 0:
        raise Exception("Image shape must be divisible by desried shape.", image.shape, desired_shape)

    patches = []
    ds = desired_shape

    num_cols = image.shape[0] / desired_shape[0]
    num_rows = image.shape[1] / desired_shape[1]

    for x_idx in range(num_rows):
        for y_idx in range(num_cols):
            start_x = x_idx * ds[0]
            end_x = (x_idx+1) * ds[0]
            start_y = y_idx * ds[1]
            end_y = (y_idx+1) * ds[1]

            if len(desired_shape) == 3:
                patches.append(image[start_x:end_x, start_y:end_y, 0:ds[-1]])
            else:
                patches.append(image[start_x:end_x, start_y:end_y])

    expected_length = (image.shape[0] / desired_shape[0])**2
    if len(patches) != expected_length:
        raise Exception("Length of patches is unexpected. Expected len: {}, actual {}".format(expected_length. len(patches)))

    return np.array(patches)







def frac_eq_to(image, value=0):
    return (image == value).sum() / float(np.prod(image.shape))


# def extract_patches(image, patchshape, overlap_allowed=0.5, cropvalue=None,
#                     crop_fraction_allowed=0.1):
#     """
#     Given an image, extract patches of a given shape with a certain
#     amount of allowed overlap between patches, using a heuristic to
#     ensure maximum coverage.

#     If cropvalue is specified, it is treated as a flag denoting a pixel
#     that has been cropped. Patch will be rejected if it has more than
#     crop_fraction_allowed * prod(patchshape) pixels equal to cropvalue.
#     Likewise, patches will be rejected for having more overlap_allowed
#     fraction of their pixels contained in a patch already selected.
#     """
#     jump_cols = int(patchshape[1] * overlap_allowed)
#     jump_rows = int(patchshape[0] * overlap_allowed)

#     # Restrict ourselves to the rectangle containing non-cropped pixels
#     if cropvalue is not None:
#         rows, cols = np.where(image != cropvalue)
#         rows.sort()
#         cols.sort()
#         active = image[rows[0]:rows[-1], cols[0]:cols[-1]]
#     else:
#         active = image

#     rowstart = 0
#     colstart = 0

#     # Array tracking where we've already taken patches.
#     covered = np.zeros(active.shape, dtype=bool)
#     patches = []

#     while rowstart < active.shape[0] - patchshape[0]:
#         # Record whether or not e've found a patch in this row,
#         # so we know whether to skip ahead.
#         got_a_patch_this_row = False
#         colstart = 0
#         while colstart < active.shape[1] - patchshape[1]:
#             # Slice tuple indexing the region of our proposed patch
#             region = (slice(rowstart, rowstart + patchshape[0]),
#                       slice(colstart, colstart + patchshape[1]))

#             # The actual pixels in that region.
#             patch = active[region]

#             # The current mask value for that region.
#             cover_p = covered[region]
#             if cropvalue is None or \
#                frac_eq_to(patch, cropvalue) <= crop_fraction_allowed and \
#                frac_eq_to(cover_p, True) <= overlap_allowed:
#                 # Accept the patch.
#                 patches.append(patch)

#                 # Mask the area.
#                 covered[region] = True

#                 # Jump ahead in the x direction.
#                 colstart += jump_cols
#                 got_a_patch_this_row = True
#                 # print "Got a patch at %d, %d" % (rowstart, colstart)
#             else:
#                 # Otherwise, shift window across by one pixel.
#                 colstart += 1

#         if got_a_patch_this_row:
#             # Jump ahead in the y direction.
#             rowstart += jump_rows
#         else:
#             # Otherwise, shift the window down by one pixel.
#             rowstart += 1

#     # Return a 3D array of the patches with the patch index as the first
#     # dimension (so that patch pixels stay contiguous in memory, in a
#     # C-ordered array).
#     return np.concatenate([pat[np.newaxis, ...] for pat in patches], axis=0)


def plot_patches(patches, fignum=None, low=0, high=0):
    """
    Given a stack of 2D patches indexed by the first dimension, plot the
    patches in subplots.

    'low' and 'high' are optional arguments to control which patches
    actually get plotted. 'fignum' chooses the figure to plot in.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(fignum)
    if high == 0:
        high = len(patches)
    pmin, pmax = patches.min(), patches.max()
    dims = np.ceil(np.sqrt(high - low))
    for idx in xrange(high - low):
        plt.subplot(dims, dims, idx + 1)
        plt.axis('off')
        plt.imshow(patches[idx], cmap=matplotlib.cm.gray)
        plt.clim(pmin, pmax)
    plt.show()
    plt.pause(0.001)


def neuron_patches(patches, min_mean=0.0, min_std=0.0):
    """
    neuron patches by some criterion on their mean and variance.

    Takes patches, a 3-dimensional stack of image patches (where
    the first dimension indexes the patch), and a minimum
    mean and standard deviation. Returns a stack of all the 
    patches that satisfy both of these criteria.
    """
    patchdim = np.prod(patches.shape[1:])
    patchvectors = patches.reshape(patches.shape[0], patchdim)
    means = patchvectors.mean(axis=1)
    stdevs = patchvectors.std(axis=1)
    indices = (means > min_mean) & (stdevs > min_std)
    return patches[indices]


def extract_patches_from_dir(directory, patchsize,
                             smoothing=None, overlap_allowed=0.5,
                             cropvalue=None, crop_fraction_allowed=0.1,
                             min_mean=0, min_std=0):
    """
    Extract patches from an entire directory of images.

    If `smoothing` is not None, it is used as the standard deviation of a
    Gaussian neuron applied to the image before extracting patches.

    `patchsize`, `overlap_allowed`, `cropvalue` and `crop_fraction_allowed`
    are passed along to `extract_patches()`. `min_mean` and `min_std` are
    passed along to `neuron_patches()`.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    output = {}
    for fname in os.listdir(directory):
        if fname[-4:] == '.png':
            outname = fname.replace('.', '_').replace('-', '_')
            assert outname not in output
            image = plt.imread(os.path.join(directory, fname))
            if smoothing is not None:
                image = ndimage.gaussian_neuron(image, smoothing)
            # Extract patches from the image.
            output[outname] = extract_patches(image, patchsize,
                                              overlap_allowed,
                                              cropvalue, crop_fraction_allowed)

            # neuron the patches that don't meet our standards.
            output[outname] = neuron_patches(output[outname], min_std=min_std,
                                             min_mean=min_mean)
    return output
