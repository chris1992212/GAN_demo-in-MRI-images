import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import scipy.io as sio
from tqdm import tqdm
import numpy as np
import time
import os
import utils

class AI_GAN_DEMO(object):
    def __init__(self,
                 sess,
                 image_size_FE,
                 image_size_PE,
                 batch_size,
                 c_dim,
                 test_FE,
                 test_PE):
        """

        :param sess: open a session for training
        :param image_size_FE: Size of training data in x direction (after data augmentation: cropping)
        :param image_size_PE: Size of training data in y direction
        :param batch_size: Number of training samples every iteration
        :param c_dim: To determine the fourth channel of Data, used for multi-contrast image or complex image;(for example, data_size: batch_size*image_size_FE*image_PE*c_dim)
        :param test_FE: Size of testing data in x direction (without augmention)
        :param test_PE: Size of testing data in y direction (without augmention)
        """
        self.sess = sess
        self.image_size_FE = image_size_FE
        self.image_size_PE = image_size_PE
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.test_FE = test_FE
        self.test_PE = test_PE

    def tfrecord_read_dataset(self, batch_size, size_FE, size_PE, Filenames, c_dim, training):
        """
        read binary data from filenames
        Argument:
            batch_size:
            size_FE: FE Num of data
            size_PE: PE Num of data
            Filenames: name of tfrecord
            c_dim: channel
            training: whether shuffle or not
        return: batch_x, batch_y
        """

        def parser(record):
            features = tf.parse_single_example(record,
                                               features={
                                                   'low_CompI': tf.FixedLenFeature([], tf.string),
                                                   'CompI': tf.FixedLenFeature([], tf.string)
                                               })
            low = tf.decode_raw(features['low_CompI'], tf.float32)
            low = tf.reshape(low, [crop_patch_FE, crop_patch_PE, Num_CHANNELS])

            high = tf.decode_raw(features['CompI'], tf.float32)
            high = tf.reshape(high, [crop_patch_FE, crop_patch_PE, Num_CHANNELS])
            return low, high

        crop_patch_FE = size_FE
        crop_patch_PE = size_PE
        Num_CHANNELS = c_dim
        batch_size = batch_size
        if training == True:
            buffer_size = 20000
        else:
            buffer_size = 1
        # output file name string to a queue
        dataset = tf.data.TFRecordDataset(Filenames)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        itertor = dataset.make_one_shot_iterator()
        final = itertor.get_next()
        return final

    def log_record(self,config):
        """
        To record the loss value of testing data during training
        :param config:
        :return:
        """
        log_dir = "log_{}".format('AI_GAN')
        tl.files.exists_or_mkdir(log_dir)
        self.log_all, self.log_all_filename = utils.logging_setup(log_dir)
        utils.log_config(self.log_all_filename, config)

    def loss_SSIM(self, y_true, y_pred):
        """
        define loss function between Ground_truth and pred of network
        :param y_pred:
        :return: losss
        """
        ssim = tf.image.ssim(y_true, y_pred, max_val=1)
        return tf.reduce_mean((1.0 - ssim) / 2, name='ssim_loss')

    def Gen_loss(self, logits_fake, net_g, label):
        """

        :param logits_real: To determine whether the ground truth is true;
        :param logits_fake: To determine whether the outputs of network is fake;
        :param net_g: Outputs of generate network
        :param label: Ground Truth
        :return:
        """
        # generator loss (adversarial)
        g_adv_loss = tf.div(tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g'),15)
        g_ssim_loss = self.loss_SSIM(net_g, label)
        g_loss_combine = g_ssim_loss + g_adv_loss
        return g_adv_loss, g_ssim_loss, g_loss_combine

    def Dis_loss(self, logits_real, logits_fake):
        """

        :param logits_real: To determine whether the ground truth is true;
        :param logits_fake: To determine whether the outputs of network is fake;
        :param net_g: Outputs of generate network
        :param label: Ground Truth
        :return:
        """
        # discriminator loss
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')# logits_real=1
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        d_loss = d_loss1 + d_loss2

        return d_loss, d_loss1, d_loss2

    def gen_model(self, images, is_train = False, reuse = False):
        """

        :param images: input of model, always undersampled images
        :param is_train: to determine whether Batch Normalization
        :param reuse: reuse name of net
        :return:
        """
        n_out = self.c_dim
        x = images
        _, nx, ny, nz = x.get_shape().as_list()

        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1, 0.02)

        with tf.variable_scope("u_net", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            inputs = tl.layers.InputLayer(x, name='inputs')

            conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_1')
            conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv1_2')
            conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                                   name='bn1')
            pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')


            conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_1')
            conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv2_2')
            conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                                   name='bn2')
            pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')



            conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_1')
            conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv3_2')
            conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                                   name='bn3')
            pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')


            conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_1')
            conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv4_2')
            conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                                   name='bn4')
            pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')


            conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_1')
            conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='conv5_2')
            conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init,
                                   name='bn5')


            print(" * After conv: %s" % conv5.outputs)

            up4 = tl.layers.DeConv2d(conv5, 512, (3, 3),
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 8)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 8))],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv4')
            up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn4_1')
            up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
            conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv4_1')
            conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv4_2')
            conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn4_2')


            up3 = tl.layers.DeConv2d(conv4, 256, (3, 3),
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 4)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 4))],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv3')
            up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn3_1')
            up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
            conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv3_1')
            conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv3_2')
            conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn3_2')


            up2 = tl.layers.DeConv2d(conv3, 128, (3, 3),
                                     out_size=[tf.to_int32(tf.ceil(tf.shape(x)[1] / 2)), tf.to_int32(tf.ceil(tf.shape(x)[2] / 2))],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv2')
            up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn2_1')
            up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
            conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv2_1')
            conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv2_2')
            conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn2_2')


            up1 = tl.layers.DeConv2d(conv2, 64, (3, 3),
                                     out_size=[tf.to_int32(tf.shape(x)[1]), tf.to_int32(tf.shape(x)[2])],
                                     strides=(2, 2),
                                     padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv1')
            up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init,
                                 name='ucov_bn1_1')
            up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
            conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='uconv1_1')
            conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                     b_init=b_init,
                                     name='uconv1_2')
            conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                                   name='ucov_bn1_2')

            conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')

            out = tf.add(conv1.outputs, inputs.outputs, name='output')
            # input = inputs.outputs
            ######## -------------------------Data fidelity--------------------------------##########
            # for contrast in range(n_out):
            #     k_conv3 = utils.Fourier(conv1[:,:,:,contrast], separate_complex=False)
            #     mask = np.ones((batch_size, nx, ny))
            #     mask[:,:, 1:ny:3] = 0
            #     mask = np.fft.ifftshift(mask)
            #     # convert to complex tf tensor
            #     DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
            #     DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
            #     k_patches = utils.Fourier(input[:,:,:,contrast], separate_complex=False)
            #     k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches*(1-DEFAULT_MAKS_TF_c)
            #     out = tf.ifft2d(k_space)
            #     out = tf.abs(out)
            #     out = tf.reshape(out, [batch_size, nx, ny, 1])
            #     if contrast == 0 :
            #         final_output = out
            #     else:
            #         final_output = tf.concat([final_output,out],3)
            ########-------------------------end------------------------------------###########3
            # print(" * Output: %s" % conv1.outputs)
            # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
            return out

    # def gen_model(self, images, is_train = False, reuse=False):
    #     """
    #     The network including three convolution layers
    #     :param images: input of network
    #     :param is_train:  To determine the parameters of Batch Normalization, no use here
    #     :param reuse: if true, reuse the name of network layers
    #     :return:
    #     """
    #     # ============= To initialize the parameters of network layers ==========#
    #     # ============  w_int, b_init: initial value of weight and bias of convontion kernels in network====== #
    #     w_init = tf.truncated_normal_initializer(stddev=0.01)# stddev can be changed for different problems
    #     b_init = tf.constant_initializer(value=0.0) # value usually 0
    #
    #     with tf.variable_scope('srcnn', reuse=reuse): # define scope names of tensors, making thing convenient in monitoring tools, like tensorboard
    #         tl.layers.set_name_reuse(reuse) # reuse name when try to validate during training, True: validation
    #         inputs = tl.layers.InputLayer(images, name='inputs')
    #         conv1 = tl.layers.Conv2d(inputs, 64, (9, 9), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
    #                                  name='conv1') # convolution layers:   kernel size 9*9, output_channel 64; activation function: relu
    #         conv2 = tl.layers.Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
    #                                  name='conv2') # convolution layers:   kernel size 3*3, output_channel 32; activation function: relu
    #         conv3 = tl.layers.Conv2d(conv2, self.c_dim, (5, 5), act=None, padding='SAME', W_init=w_init, b_init=b_init,
    #                                  name='conv3') # convolution layers:   kernel size 5*5, output_channel 1; activation function: None
    #
    #         return conv3.outputs
    def discriminator(self, input_images, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        gamma_init = tf.random_normal_initializer(1., 0.02)
        df_dim = 64

        with tf.variable_scope("discriminator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_in = InputLayer(input_images,
                                name='input')

            net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                            padding='SAME', W_init=w_init, name='h0/conv2d')

            net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h1/conv2d')
            net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h1/batchnorm')

            net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h2/conv2d')
            net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h2/batchnorm')

            net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h3/conv2d')
            net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h3/batchnorm')

            net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h4/conv2d')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h4/batchnorm')

            net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h5/conv2d')
            net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h5/batchnorm')

            net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h6/conv2d')
            net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
                                    is_train=is_train, gamma_init=gamma_init, name='h6/batchnorm')

            net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None,
                            padding='SAME', W_init=w_init, b_init=b_init, name='h7/conv2d')
            net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/batchnorm')

            net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d')
            net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm')
            net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d2')
            net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm2')
            net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d3')
            net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm3')

            net_h8 = ElementwiseLayer(layer=[net_h7, net], combine_fn=tf.add, name='h8/add')
            net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

            net_ho = FlattenLayer(net_h8, name='output/flatten')
            net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense')
            logits = net_ho.outputs
            net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

        return net_ho, logits

    # def discriminator(self, input_images, is_train=True, reuse=False):
    #     w_init = tf.random_normal_initializer(stddev=0.02)
    #     b_init = None
    #     gamma_init = tf.random_normal_initializer(1., 0.02)
    #     df_dim = 64
    #
    #     with tf.variable_scope("discriminator", reuse=reuse):
    #         tl.layers.set_name_reuse(reuse)
    #
    #         net_in = InputLayer(input_images,
    #                             name='inputs')
    #
    #         net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
    #                         padding='SAME', W_init=w_init, name='h0/conv2d')
    #
    #         net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h1/conv2d')
    #         net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h1/batchnorm')
    #
    #         net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h2/conv2d')
    #         net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h2/batchnorm')
    #
    #         net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h3/conv2d')
    #         net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h3/batchnorm')
    #
    #         net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h4/conv2d')
    #         net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h4/batchnorm')
    #
    #         net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h5/conv2d')
    #         net_h5 = BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h5/batchnorm')
    #
    #         net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h6/conv2d')
    #         net_h6 = BatchNormLayer(net_h6, act=lambda x: tl.act.lrelu(x, 0.2),
    #                                 is_train=is_train, gamma_init=gamma_init, name='h6/batchnorm')
    #
    #         net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None,
    #                         padding='SAME', W_init=w_init, b_init=b_init, name='h7/conv2d')
    #         net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/batchnorm')
    #
    #         net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None,
    #                      padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d')
    #         net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
    #                              is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm')
    #         net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None,
    #                      padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d2')
    #         net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
    #                              is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm2')
    #         net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None,
    #                      padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d3')
    #         net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm3')
    #
    #         net_h8 = ElementwiseLayer(layer=[net_h7, net], combine_fn=tf.add, name='h8/add')
    #         net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
    #
    #         net_ho = FlattenLayer(net_h8, name='output/flatten')
    #         net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense')
    #         logits = net_ho.outputs
    #         net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    #
    #     return net_ho, logits

    def build_model(self,config):
        """
        To define the placeholder and loss function during the whole training
        1) To define the placeholder for input and output of neural network;
        2) To define loss functions for training and testing, validating
        :return:
        """
        #======================== define placeholer =================================#
        self.images = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='labels')
        self.validation_images = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_images')
        self.validation_labels = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_labels')

        #========================= tensor: output of the generative network ====================#
        self.pred_gen = self.gen_model(self.images, is_train = True, reuse = False) # training
        self.validation_pred_gen = self.gen_model(self.validation_images, is_train = False, reuse= True) # validation

        #========================= tensor: output of the adversarial network ====================#
        _, self.train_logits_fake = self.discriminator(self.pred_gen, is_train=True, reuse=False) # training: discrim for the output of generative network
        _, self.train_logits_real = self.discriminator(self.labels, is_train=True, reuse=True) # discrim for the label

        # _, validation_logits_fake = self.discriminator(self.validation_pred_gen, is_train=False, reuse=True)  # validation
        # _, validation_logits_real = self.discriminator(self.validation_labels, is_train=False, reuse=True)

        #========================== Loss function caclucation =============== #
              #####===================== Generative loss ==================#####
        self.g_adv_loss, self.g_ssim_loss, self.preding_gen_loss = self.Gen_loss(self.train_logits_fake, self.pred_gen, self.labels)# Loss between output of net with ground_truth for training
        # _, self.srcing_gen_loss = self.Gen_loss(train_logits_fake, self.images, self.labels)# Loss between input of net with ground_truth for training
        self.validation_preding_gen_loss = self.loss_SSIM(self.validation_pred_gen, self.validation_labels) # Loss between output of net with ground_truth for validation
        self.validation_srcing_gen_loss = self.loss_SSIM( self.validation_images, self.validation_labels) # Loss between input of net with ground_truth for validation
             #####==================== Discriminatory loss ================#####
        self.preding_dis_loss ,self.d_loss1, self.d_loss2 = self.Dis_loss(self.train_logits_real, self.train_logits_fake)  # Loss between output of net with ground_truth for training

        #========================= define the option during training================#
        self.saver = tf.train.Saver() # save model option

        #========================= Define vars of Generative and Discriminatory Network============#
        g_vars = tl.layers.get_variables_with_name('u_net', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        #============ define the optimization of neural network =============#
        self.train_op_gen = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.preding_gen_loss, var_list=g_vars)
        self.train_op_dis = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.preding_dis_loss, var_list=d_vars)

    def train(self, config):
        """
       Load data and begin train
        :param config:
        :return:
        """
        #============================== Load Data ==================================================#
        train_data = self.tfrecord_read_dataset(self.batch_size,config.image_size_FE,config.image_size_PE,config.tfrecord_train,self.c_dim, True)
        batch_valid = self.tfrecord_read_dataset(1,config.test_FE,config.test_PE,config.tfrecord_test,self.c_dim, False)

        #=================================== initlize variables and restore model=================#
        tf.global_variables_initializer().run()
        self.saver.restore(self.sess, config.save_model_filename)

        #============================== save log to txt =============================#
        self.log_record(config)  # save log to txt

        #====================================== define array of loss values===========================#
        test_src_loss = np.zeros((config.TESTING_NUM, 1))
        test_pred_loss = np.zeros((config.TESTING_NUM, 1))
        train_src_loss = np.zeros((config.TESTING_NUM, 1))
        train_pred_loss = np.zeros((config.TESTING_NUM, 1))
        # =======================================Training Cycle=====================================#
        if True:
            print("Now Start Training...")
            best_mse = 1
            early_stop_number =100
            for epoch in tqdm(range(config.epoch)):
                # =================Run by batch images =======================#
                batch_xs, batch_ys = self.sess.run(train_data) # Get training data for training

                # ============== training =======================#
                ## ==== train_op: can calculate the descent gradient for next epoch
                # ====== preding_loss: printf to see the loss
                # ===== out: output of current net parameters
                # ============== Op_g, Op_dis

                err_dis, err_gen, _ = self.sess.run([self.preding_dis_loss, self.g_adv_loss, self.train_op_dis], feed_dict = {self.images: batch_xs, self.labels: batch_ys})

                err_ssim_gen, err_gen, gen_net, _ = self.sess.run([ self.g_ssim_loss, self.g_adv_loss, self.pred_gen, self.train_op_gen], feed_dict = {self.images: batch_xs, self.labels: batch_ys})

                # ========================= Save model ===============#
                if epoch % 100 == 0:
                    print('epoch %d Adversial cost => %.7f and Discriminatory cost => %.7f' % (epoch, err_gen, err_dis))
                    print('SSIM loss of Generator net => %.7f' %(err_ssim_gen))
                    # save_path = self.saver.save (self.sess, config.save_model_filename)
                if epoch % 1000 == 0:
                    save_path = self.saver.save (self.sess, config.save_model_filename)

                    # self.saver.restore(self.sess, config.save_model_filename)
                    for ep in range(config.TESTING_NUM):
                        # batch_xs_validation, batch_ys_validation = next(test_data)
                        batch_xs_validation, batch_ys_validation = self.sess.run(batch_valid)
                        test_src_loss[ep], test_pred_loss[ep] = self.sess.run([self.validation_srcing_gen_loss, self.validation_preding_gen_loss],
                                                                       feed_dict = { self.validation_images: batch_xs_validation, self.validation_labels: batch_ys_validation})
                    print( 'epoch: %d ,ave_src_SSIM: %.7f,ave_pred_SSIM: %.7f' % ( epoch, test_src_loss.mean(), test_pred_loss.mean()))
                    if test_pred_loss.mean() < best_mse:
                        save_path = self.saver.save(self.sess, config.save_model_filename_best)
                        best_mse = test_pred_loss.mean()
                        ear_stop = early_stop_number
                        best_epoch = epoch
                    else:
                        ear_stop -= 1
                    print('best_epoch: %d, ear_stop: %d' % (best_epoch, ear_stop))
                    if ear_stop == 0:
                        print('best_mse: %.7f' % (best_mse))
                        break
                    log = "Best_epoch: {}\n Epoch: {}\n mse val: {:8}\n mse_src: {:8}\n ".format(
                    best_epoch,
                    epoch + 1,
                    test_pred_loss.mean(),
                    test_src_loss.mean(),
                    )
                    print(log)
                    self.log_all.debug(log)

        self.sess.close() # close the session

    def pred_test(self, config):
        #======= Get testing data (tensor)  and restore model=============#
        batch_valid = self.tfrecord_read_dataset(1,config.test_FE,config.test_PE,config.tfrecord_test,self.c_dim,False)
        self.saver.restore(self.sess, config.save_model_filename) # restore model parameters saved during training

        #============= Initialize array for caculating MSE value==============#
        test_src_mse = np.zeros((config.TESTING_NUM, 1))
        test_pred_mse = np.zeros((config.TESTING_NUM, 1))
        recon = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        high_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        low_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')

        #============== Begin Testing===================#
        for ep in range(config.TESTING_NUM):
            batch_xs_validation, batch_ys_validation = self.sess.run(batch_valid) # Get testing data for every iteration

            recon[:,:,ep,:], high_res_images[:,:,ep,:], low_res_images[:,:,ep,:] = self.sess.run([self.validation_pred_gen, self.validation_labels, self.validation_images],
                                                              feed_dict={self.validation_images: batch_xs_validation,
                                                                         self.validation_labels: batch_ys_validation}) # get network output, ground_truth, input of network

            test_src_mse[ep], test_pred_mse[ep] = self.sess.run(
                [self.validation_srcing_gen_loss, self.validation_preding_gen_loss],
                feed_dict={self.validation_images: batch_xs_validation,
                           self.validation_labels: batch_ys_validation}) # calculate the loss between  network output with ground_truth, input with ground_truth
            print('ave_src_MSE: %.7f,ave_pred_MSE: %.7f' % (test_src_mse[ep], test_pred_mse[ep]))
        print('mean ave_src_MSE: %.7f,mean ave_pred_MSE: %.7f' % (test_src_mse.mean(), test_pred_mse.mean()))
        saving_path = 'F:\matlab\Data_address_cml\BrainQuant_AI\Qin_Data\\finetune'
        # saving_path = 'F:\matlab\Data_address_cml\BrainQuant_AI\Qin_Data\XUWEI'

        tl.files.exists_or_mkdir(saving_path)
        sio.savemat(os.path.join(saving_path, 'recon_3channel_xuwei_SSIM.mat'), {'recon_3channel_SSIM': recon})
        sio.savemat(os.path.join(saving_path,'low_res_images.mat'), {'low_res_images': low_res_images})
        sio.savemat(os.path.join(saving_path,'high_res_images.mat'), {'high_res_images': high_res_images})

        self.sess.close()