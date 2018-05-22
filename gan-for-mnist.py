import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Read the dataset (MNIST)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Discriminator
def discriminator(images, reuse_variables=None):
	with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
		# First convolutional and pooling layers
		# This finds 32 different 5 x 5 pixel features
		d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
		d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
		d1 = d1 + d_b1
		d1 = tf.nn.relu(d1)
		d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		# Second convolutional and pooling layers
		# This finds 64 different 5 x 5 pixel features
		d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
		d2 = d2 + d_b2
		d2 = tf.nn.relu(d2)
		d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		# First fully connected layer
		d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
		d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
		d3 = tf.matmul(d3, d_w3)
		d3 = d3 + d_b3
		d3 = tf.nn.relu(d3)

		# Second fully connected layer
		d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
		d4 = tf.matmul(d3, d_w4) + d_b4

		# d4 contains unscaled values
		return d4

# Generator
def generator(z, batch_size, z_dim):
	# From z_dim to 7 * 7 * 64 dimension
	g_w1 = tf.get_variable('g_w1', [z_dim, 7 * 7 * 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b1 = tf.get_variable('g_b1', [7 * 7 * 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g1 = tf.matmul(z, g_w1) + g_b1
	g1 = tf.reshape(g1, [-1, 7, 7, 64])
	g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
	g1 = tf.nn.relu(g1)
	b_size = tf.shape(g1)[0]

	# Deconv1
	g_w2 = tf.get_variable('g_w2', [5, 5, 32, 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b2 = tf.get_variable('g_b2', [32], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g2 = tf.nn.conv2d_transpose(g1, g_w2, [b_size, 14, 14, 32], strides=[1, 2, 2, 1], padding='SAME')
	g2 = g2 + g_b2
	g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
	g2 = tf.nn.relu(g2)

	# Deconv2
	g_w3 = tf.get_variable('g_w3', [5, 5, 16, 32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b3 = tf.get_variable('g_b3', [16], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g3 = tf.nn.conv2d_transpose(g2, g_w3, [b_size, 28, 28, 16], strides=[1, 2, 2, 1], padding='SAME')
	g3 = g3 + g_b3
	g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
	g3 = tf.nn.relu(g3)

	# Deconv3
	g_w4 = tf.get_variable('g_w4', [5, 5, 1, 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g4 = tf.nn.conv2d_transpose(g3, g_w4, [b_size, 28, 28, 1], strides=[1, 1, 1, 1], padding='SAME')
	g4 = g4 + g_b4
	g4 = tf.sigmoid(g4)

	# Dimension of g4: batch_size 28 x 28 x 1
	return g4


""" See the fake image we make """
# Define the placeholder and the graph
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

def generate_img():
	# For generator, one image for a batch
	generated_image_output = generator(z_placeholder, 1, z_dimensions)
	z_batch = np.random.normal(0, 1, [1, z_dimensions])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		generated_image = sess.run(generated_image_output,
									feed_dict={z_placeholder: z_batch})
		generated_image = generated_image.reshape([28, 28])
		plt.imshow(generated_image, cmap='Greys')
		plt.savefig("./img/test_img.png")


""" inputs and outputs """
batch_size = 50
# z_placeholder is for feeding input noise to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')

# x_placeholder is for feeding input images to the discriminator
x_placeholder = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name='x_placeholder')

# Gz holds the generated images
# Gz = generator(z_placeholder, batch_size, z_dimensions)
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Dx will hold discriminator prediction probabilities for the real MNIST images
Dx = discriminator(x_placeholder)

# Dg will hold discriminator prediction probabilities for generated images
Dg = discriminator(Gz, reuse_variables=True)


""" Loss functions """
# Two Loss Functions for discriminator
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

# Loss function for generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))


""" Variables """
# Get the varaibles for different network
tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]


""" Train (Optimizer) """
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)



""" Start Training Session """
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
print('Pre-train')
for i in range(500):
	z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
	real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
	_, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
										   {x_placeholder: real_image_batch, z_placeholder: z_batch})

	if(i % 100 == 0):
		print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(10000):
	real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
	z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

	# Train discriminator on both real and fake images
	_, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
										   {x_placeholder: real_image_batch, z_placeholder: z_batch})

	# Train generator
	for j in range(2):
		z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
		_, gLoss = sess.run([g_trainer, g_loss], feed_dict={z_placeholder: z_batch})

	if(i % 50 == 0):
		print('iter', i, "dLossReal:", dLossReal, "dLossFake:", dLossFake, "gLoss:", gLoss)

		# generate image & save
		z_batch = np.random.normal(0, 1, [9, z_dimensions])
		generated_image = sess.run(Gz, feed_dict={z_placeholder: z_batch})
		generated_image = generated_image.reshape([9, 28, 28])
		plt.figure()
		for k in range(9):
			plt.subplot(3, 3, k+1)
			plt.imshow(generated_image[k], cmap='Greys')
		plt.savefig('./img0522_4/gen_img' + str(i) + '.png')



