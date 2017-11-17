"""
Implementation of Capsule Network
https://arxiv.org/pdf/1710.09829.pdf
Author: Ayan Das
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, shutil
from numpy import zeros, float32

bsize = 128
epochs = 100
MPlus = 0.9
MMinus = 0.1
lam = 0.5
eps = 1e-15
reco_loss_importance = 5e-4 * 784
routing_iter = 3

def squash(x, axis):
    # x: input tensor
    # axis: which axis to squash
    # I didn't use tf.norm() here to avoid mathamatical instability
    sq_norm = tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)
    scalar_factor = sq_norm / (1 + sq_norm) / tf.sqrt(sq_norm + eps)
    return tf.multiply(scalar_factor, x)

def main( args=None ):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # X, Y = get_mini_mnist(bsize=bsize, as_image=True)

    image_w, image_h, image_c = 28, 28, 1
    conv1_kernel_size, conv1_fea_maps, conv1_strides = 9, 256, 1
    convcaps_kernel_size, convcaps_stride = 9, 2
    convcaps_capsule_dim, convcaps_fea_maps = 8, 32
    n_class, out_capsule_dim = 10, 16
    reco_net1, reco_net2, reco_net3 = 512, 1024, image_h*image_w

    with tf.name_scope('phs') as phs: # all the placeholders
        x = tf.placeholder(tf.float32, shape=(None,image_w,image_h,image_c), name='x')
        y = tf.placeholder(tf.float32, shape=(None,n_class), name='y')

    with tf.name_scope('conv1'):
        conv1_act = tf.layers.conv2d(x, filters=conv1_fea_maps, kernel_size=conv1_kernel_size, strides=conv1_strides, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
            name='conv1', padding='VALID', use_bias=True, bias_initializer=tf.initializers.zeros)

    with tf.name_scope('PrimCaps'):
        primecaps_act = tf.layers.conv2d(conv1_act, filters=convcaps_capsule_dim*convcaps_fea_maps,
            kernel_size=convcaps_kernel_size, strides=convcaps_stride,
            kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
            name='primecaps', padding='VALID', use_bias=True, bias_initializer=tf.initializers.zeros)
            # (B x 6 x 6 x 256)

        # reshape it as a layer of 1152 capsules
        primecaps_act = tf.reshape(primecaps_act, shape=(bsize, 6*6*convcaps_fea_maps, 1, convcaps_capsule_dim, 1)) 
            # (B x 1152 x 1 x 8 x 1)

        # tiling it for further simplicity of multiplication (with W)
        primecaps_act = tf.tile(primecaps_act, [1,1,n_class,1,1]) # (B x 1152 x 10 x 8 x 1)
        # they are capsules, so squashing is required
        primecaps_act = squash(primecaps_act, axis=3) # same

    with tf.variable_scope('params') as params:
        with tf.variable_scope('Caps2Caps'):
            # the caps-to-caps weight tensor
            W = tf.get_variable('W', dtype=tf.float32, # (1, 1152, 10, 8, 16)
                initializer=tf.initializers.random_normal(stddev=0.1),
                shape=(1, 6*6*convcaps_fea_maps, n_class, convcaps_capsule_dim, out_capsule_dim))

            # tiling it also for simplicity of multiplication
            W = tf.tile(W, multiples=[bsize,1,1,1,1], name='tile_W') # (B x 1152 x 10 x 8 x 16)

    with tf.name_scope('Caps2Digs'):
        # this is the entire 'u_hat_j|i' in the paper
        u = tf.matmul(W, primecaps_act, transpose_a=True) # (B x 1152 x 10 x 16 x 1)
        # reshape it for routing
        u = tf.reshape(tf.squeeze(u), shape=(-1, 6*6*convcaps_fea_maps, out_capsule_dim, n_class),
        name='u') # (B x 1152 x 16 x 10)

        # the logits (b_ij); this is a constant so it persists across for_loop in the graph
        # it re-inits in the next batch (i.e. in the next sess.run() call)
        bij = tf.constant(zeros((bsize, 6*6*convcaps_fea_maps, n_class), dtype=float32),
            dtype=tf.float32, name='bij') # (B x 1152 x 10)

        # the routing iteration
        for route_iter in range(routing_iter):
            with tf.name_scope('route_' + str(route_iter)):
                # making sure sum_cij_over_j is one
                cij = tf.nn.softmax(bij, dim=2) # (B x 1152 x 10)

                # the 's_j' in the paper
                s = tf.reduce_sum(u * tf.reshape(cij, shape=(bsize,6*6*convcaps_fea_maps,1,n_class)),
                    axis=1, keep_dims=False) # (B x 16 x 10)

                # v_j = squash(s_j) from the paper
                v = squash(s, axis=1) # (B x 16 x 10)

                if route_iter < routing_iter - 1: # bij computation not required at the end
                    # Here comes 'routing', reshaping v for further multiplication
                    v_r = tf.reshape(v, shape=(-1, 1, out_capsule_dim, n_class)) # (B x 1 x 16 x 10)

                    # the 'agreement' in the paper
                    uv_dot = tf.reduce_sum(u*v_r, axis=2, name='uv')

                    # update logits with the agreement
                    bij += uv_dot

    with tf.name_scope('loss'):

        # reconstruction loss
        with tf.name_scope('recon_loss'):
            with tf.variable_scope(params):
                layer1 = tf.layers.Dense(reco_net1, activation=tf.nn.relu, use_bias=True,
                    kernel_initializer=tf.initializers.random_normal)
                layer2 = tf.layers.Dense(reco_net2, activation=tf.nn.relu, use_bias=True,
                    kernel_initializer=tf.initializers.random_normal)
                layer3 = tf.layers.Dense(reco_net3, activation=tf.nn.sigmoid, use_bias=True,
                    kernel_initializer=tf.initializers.random_normal)

            v_masked = tf.multiply(v, tf.reshape(y, shape=(-1,1,n_class)))
            v_masked = tf.reduce_sum(v_masked, axis=2, name='v_mask') # (B x 16)

            # the entire reconstruction network
            v_reco = layer3( layer2( layer1( v_masked ) ) )

            reco_loss = tf.reduce_mean(tf.square(v_reco - tf.reshape(x, shape=(-1, 784))),
                name='reco_loss')

        # the length of v_j
        v_len = tf.sqrt(tf.reduce_sum(tf.square(v), axis=1) + eps) # (B x 10)

        # classification loss
        with tf.name_scope('classif_loss'):
            # loss as proposed in the paper
            l_klass = y * (tf.maximum(zeros((1,1),dtype=float32), MPlus-v_len)**2) + \
                lam * (1-y) * (tf.maximum(zeros((1,1),dtype=float32), v_len-MMinus)**2)

            class_loss = tf.reduce_mean(l_klass, name='loss')

        # whole loss
        with tf.name_scope('full_loss'):
            loss = class_loss + reco_loss * reco_loss_importance
            loss_summary = tf.summary.scalar('loss', loss)

    with tf.name_scope('testing'):
        correct_prediction = tf.equal(tf.argmax(v_len, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        # accuracy_summary = tf.summary.scalar('acc', accuracy)

    with tf.name_scope('optim'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("CAPSLog")
        writer.add_graph(graph=sess.graph)

        gstep = 1

        for E in range(epochs):
            for I in range(int(55000 / bsize)):
                X, Y = mnist.train.next_batch(bsize, shuffle=True)
                X = X.reshape((-1, 28, 28, 1))

                _, l, l_sum, acc_ = sess.run([train_step, loss, loss_summary, accuracy],
                    feed_dict={x: X, y: Y})
                if I % 5 == 0:
                    print('loss: {0} - acc: {1} - batch: {2}'.format(l, acc_, I))
                writer.add_summary(l_sum, gstep)
                gstep += 1
            t_accs = 0
            print('Testing on validation set')
            for I in range(int(5000 / bsize)):
                Xt, Yt = mnist.validation.next_batch(bsize)
                Xt = Xt.reshape((-1,28,28,1))
                
                acc_test = sess.run(accuracy, feed_dict={x: Xt, y: Yt})
                t_accs += acc_test
            acc_t_avg = t_accs / int(5000 / bsize)
            print('Epoch {0} test-acc {1}'.format(E, acc_t_avg))


if __name__ == '__main__':
    if os.path.exists("CAPSLog"):
        shutil.rmtree("CAPSLog")
    main()