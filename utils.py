import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os


def linear(x, n_output, name=None, activation=None, reuse=None):
	"""
	
	:param x: Input tensor to connect
	:param n_output: number of output neurons
	:param name: define scope of layers
	:param activation: what kind of activation wants to use
	:param reuse: whether wants to reuse or not
	:return: output of fully connected network
	"""
	# If shape of input tensor is two then we have to add another dimension because for convolution we need
	# four dimension
	if len(x.get_shape()) != 2:
		x = flatten(x, reuse=reuse)
	
	# get the shape of input vector
	n_input = x.get_shape().as_lit()[1]
	
	with tf.variable_scope(name or 'fc', reuse=reuse):
		W = tf.get_variable(name='W', shape=[n_input, n_output], dtype=tf.float32,
		                    initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable(name='b', shape=[n_output], dtype=tf.float32,
		                    initializer=tf.contrib.layers.xavier_initializer())
		h = tf.nn.bias_add(name='h', value=tf.matmul(x, W), bias=b)
		
		if activation:
			h = activation(h)
	
	return h, W


def flatten(x, name=None, reuse=None):
	""" flatten tensor to 2-dimensions
	
	:param x: input tensor to flatten
	:param name: name of operation
	:param reuse: for reuse purpose
	:return: flattened tensor
	"""
	global flattened
	with tf.variable_scope('flatten'):
		dims = x.get_shape().as_list()
		if len(dims) == 4:
			flattened = tf.reshape(x, shape=[-1, dims[1] * dims[2] * dims[3]])
		elif len(dims) == 2 or len(dims) == 1:
			flattened = x
		else:
			raise ValueError('Expected n dimensions of 1, 2 or 4. Found: ', len(dims))
	return flattened

def montage(images, saveto='montage.png'):
	"""
	Creates all images as a montage separeted by 1 pixel
	
	:param images: imput array to create montage of array batch X height X width X channels
	:param saveto: destination to save the montage
	:return: Montage image
	"""
	if isinstance(images, list):
		images = np.array(images)
	img_h = images.shape[1]
	img_w = images.shape[2]
	n_plots = int(np.ceil(np.sqrt(images.shape[0])))
	if len(images.shape) == 4 and images.shape[3] == 3:
		m = np.ones()
