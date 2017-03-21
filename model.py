# Now get necessary libraries
try:
	import os
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.transform import resize
	from skimage import data
	from scipy.misc import imresize
	from scipy.ndimage.filters import gaussian_filter
	import IPython.display as ipyd
	import tensorflow as tf
	from gif import build_gif
	from stylenet import get_vgg_model
	from stylenet import preprocess, deprocess
except ImportError:
	print('Some libraries are missing....')

content_og = plt.imread('content_1.jpg')[..., :3]
style_img = plt.imread('content.jpg')[..., :3]


# Write a function to display two images

def display_img(content_img, style_img):
	fig, axs = plt.subplots(1, 2)
	axs[0].imshow(content_img)
	axs[0].set_title('content image')
	axs[0].grid('Off')
	axs[1].imshow(style_img)
	axs[1].set_title('style image')
	axs[1].grid('Off')
	plt.show()
	return


# Lets preprocess or images

content_img = preprocess(content_og)[np.newaxis]
style_img = preprocess(style_img)[np.newaxis]


def make_4d(img):
	"""Create a 4-dimensional N x H x W x C image.

    Parameters
    ----------
    img : np.ndarray
        Given image as H x W x C or H x W.

    Returns
    -------
    img : np.ndarray
        N x H x W x C image.

    Raises
    ------
    ValueError
        Unexpected number of dimensions.
    """
	if img.ndim == 2:
		img = np.expand_dims(img[np.newaxis], 3)
	elif img.ndim == 3:
		img = img[np.newaxis]
	elif img.ndim == 4:
		return img
	else:
		raise ValueError('Incorrect dimensions for image!')
	return img


def stylize(content_img, style_img, base_img=None, saveto=None, gif_step=5,
            n_iterations=300, style_weight=0.8, content_weight=0.6):
	"""Stylization w/ the given content and style images.

    Follows the approach in Leon Gatys et al.

    Parameters
    ----------
    content_img : np.ndarray
        Image to use for finding the content features.
    style_img : TYPE
        Image to use for finding the style features.
    base_img : None, optional
        Image to use for the base content.  Can be noise or an existing image.
        If None, the content image will be used.
    saveto : str, optional
        Name of GIF image to write to, e.g. "stylization.gif"
    gif_step : int, optional
        Modulo of iterations to save the current stylization.
    n_iterations : int, optional
        Number of iterations to run for.
    style_weight : float, optional
        Weighting on the style features.
    content_weight : float, optional
        Weighting on the content features.

    Returns
    -------
    stylization : np.ndarray
        Final iteration of the stylization.
    """
	# Preprocess both content and style images
	global synth
	content_img = make_4d(content_img)
	style_img = make_4d(style_img)
	if base_img is None:
		base_img = content_img
	else:
		base_img = make_4d(base_img)
	
	# Get Content and Style features
	net = get_vgg_model()
	g = tf.Graph()
	with tf.Session(graph=g) as sess:
		tf.import_graph_def(net['graph_def'], name='vgg')
		names = [op.name for op in g.get_operations()]
		print(names)
		x = g.get_tensor_by_name(names[0] + ':0')
		content_layer = 'vgg/conv5_2/conv5_2:0'
		content_features = g.get_tensor_by_name(
				content_layer).eval(feed_dict={
			x: content_img,
			'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
			'vgg/dropout/random_uniform:0': [[1.0] * 4096]
		})
		style_layers = ['vgg/conv1_1/conv1_1:0',
		                'vgg/conv2_1/conv2_1:0',
		                # 'vgg/conv3_1/conv3_1:0',
		                # 'vgg/conv4_1/conv4_1:0',
		                'vgg/conv5_1/conv5_1:0']
		style_activations = []
		for style_i in style_layers:
			style_activation_i = g.get_tensor_by_name(style_i).eval(
					feed_dict={
						x: style_img,
						'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
						'vgg/dropout/random_uniform:0': [[1.0] * 4096]
					})
			style_activations.append(style_activation_i)
		style_features = []
		for style_activation_i in style_activations:
			s_i = np.reshape(style_activation_i,
			                 [-1, style_activation_i.shape[-1]])
			gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
			style_features.append(gram_matrix.astype(np.float32))
	
	# Optimize both
	g = tf.Graph()
	with tf.Session(graph=g) as sess:
		net_input = tf.Variable(base_img)
		tf.import_graph_def(
				net['graph_def'],
				name='vgg',
				input_map={'images:0': net_input})
		
		content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer) -
		                              content_features) /
		                             content_features.size)
		style_loss = np.float32(0.0)
		for style_layer_i, style_gram_i in zip(style_layers, style_features):
			layer_i = g.get_tensor_by_name(style_layer_i)
			layer_shape = layer_i.get_shape().as_list()
			layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
			layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
			gram_matrix = tf.matmul(
					tf.transpose(layer_flat), layer_flat) / layer_size
			style_loss = tf.add(
					style_loss, tf.nn.l2_loss(
							(gram_matrix - style_gram_i) /
							np.float32(style_gram_i.size)))
		loss = content_weight * content_loss + style_weight * style_loss
		optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
		
		sess.run(tf.global_variables_initializer())
		imgs = []
		for it_i in range(n_iterations):
			_, this_loss, synth = sess.run(
					[optimizer, loss, net_input],
					feed_dict={
						'vgg/dropout_1/random_uniform:0': np.ones(
								g.get_tensor_by_name(
										'vgg/dropout_1/random_uniform:0'
								).get_shape().as_list()),
						'vgg/dropout/random_uniform:0': np.ones(
								g.get_tensor_by_name(
										'vgg/dropout/random_uniform:0'
								).get_shape().as_list())
					})
			print("iteration %d, loss: %f, range: (%f - %f)" %
			      (it_i, this_loss, np.min(synth), np.max(synth)), end='\r')
			if it_i % 5 == 0:
				m = deprocess(synth[0])
				# imgs.append(m)
				plt.imshow(m)
				plt.savefig('mixed'+str(it_i) + '.png')
			if it_i % gif_step == 0:
				imgs.append(np.clip(synth[0], 0, 1))
		if saveto is not None:
			build_gif(imgs, saveto=saveto)
	return np.clip(synth[0], 0, 1)


mo = stylize(content_img=content_img, style_img=style_img, saveto='model.gif')
mo_deprocess = deprocess(mo)
plt.imshow(mo_deprocess)
plt.savefig('final_image.png')