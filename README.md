## Deep Convolutional SOM
*Exploring a novel Deep Convolutional SOM*


![Input layer SOM](https://github.com/jakobovski/deep-convolutional-SOM/raw/master/assets/layer_1_img.png "Input layer SOM" )

(notice the similarity to visual cortex orientation columns)

### What is this repository?
This repository contains the code for a deep convolutional SOM. It is an idea I thought worth exploring. See: Inspiration. 


### TLDR: Results
Inconclusive. Besides for generating the beautiful image above, the model in it's current state it is not useful for anything practical. It is possible that adding the improvements described in "What needs work" section could transform this into a useful architecture.


### Inspiration
There are a variety of inspirations for this project:
1. SGD currently requires too much data to work well.
2. The large number of inhibitory neurons running parallel to cortical surface are rather mysterious. An SOM is a possible explanation.
3. The famous cortical orientation columns can be explained quite well with a convolution SOM.


### Why its a good idea
1. It learns unsupervised with a small amount of data.
2. It is very insensitive to noise. Because higher layers are connected to a 2D point in their input layer, and input is weighted by the distance between the connection point and the activated point. This works because nearby points on an SOM layer represent similar data.
3. It is really simple, and simple is usually better. (Occam's razer)


### What needs work 
The big, big issue with this architecture is that it is essentially doing template matching. Template matching is not scalable.  The 2D linear space, as used in an SOM, is not descriptive enough. An additional 'modifier' space is needed. This is the language equivalent of adjectives, which greatly increase representational capacity.
For example:  Given X nouns and Y adjectives we can describe X * Y objects, which is far greater than  X + Y objects. If we allow multiple adjectives to modify a single noun then the gains are even greater.


### How it works.
I suggest you look at the code, it is rather readable. 

The basic idea is to take patches of the input image (Ex: 4x4px) and pass them as input to an SOM. Lets say the SOM is 12x12x(4x4) patches. The SOM will use its 144 patches to learn representation of the inputs(see screenshot above). The SOM outputs a 2D Cartesian coordinate representing the location of the patch in itself. 

Lets say the next layer consists of 12x12x(2x2) patches, where each "pixel" in the patch patch is not a traditional pixel, but takes a Cartesian coordinate as in input.  In this way one of the 144 second layer filter's patches represents 4 Cartesian coordinates of the first layer. 

Activation of a patch is calculating the euclidean distance of each input with its internal representation. The patch with the minimum total euclidean distance is activated and sends a signal to the layer above.

This continues to arbitrary depth. One note: The first layer is a bit of an exception as the input are 1D pixels instead of 2D Cartesian coordinates. 


### Getting started
1. Clone the repository
2. Rename the folder to `alt_backprop`
3. Run one of the examples


