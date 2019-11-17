#!/usr/bin/env python

import argparse
import tensorflow as tf
import os
import datetime
import os.path as osp
from fcrn import *
from matplotlib import pyplot as plt
from utils import *
import scipy.io

def train(model, img_dir, split_path):

    height = 228
    width = 304
    channels = 3
    batch_size = 4 #16
    out_height = 128
    out_width = 160
    out_channels = 1

    # Create a placeholder for the input image
    x = tf.placeholder(tf.float32, shape = (None, height, width, channels))
    y = tf.placeholder(tf.float32, shape = (None, out_height, out_width, out_channels))
    #bg = tf.placeholder(tf.float32, shape = (None, out_height, out_width, out_channels))

    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # Construct the network
    net = ResNet50UpProj({'data': x}, batch_size, keep_prob, is_training)

    display_step = 500
    epochs = 2
    num_train_images = 556
    num_valid_images = 239
    dropout = 0.5

    learning_rate = 0.001
    #pred_fg = net.get_layer_output('ConvPred')
    #pred_bg = net.get_layer_output('ConvPred2')
    pred = net.get_output()



    # Berhu Loss--------------------------------------------------------------------------
    diff =  tf.abs(pred - y)
    valid = tf.where(tf.greater(y, tf.constant(0.0)) , tf.ones_like(y), tf.zeros_like(y))
    maxx = tf.reduce_max(diff)
    delta = tf.scalar_mul(0.2, maxx)
    l2ind = tf.where(tf.greater(diff, delta) , tf.ones_like(diff, dtype = 'int32'), tf.zeros_like(diff, dtype = 'int32'))
    diff = tf.where(tf.equal(l2ind, tf.constant(1)) , 0.5*((diff**2/delta) + delta), diff)
    diff = tf.multiply(valid, diff)
    n_valid = tf.reduce_sum(valid)
    sum_diff = tf.reduce_sum(diff)
    cost = tf.math.divide(sum_diff, n_valid)
    #-------------------------------------------------------------------------------------
    train_summary = tf.summary.scalar('training loss', cost)
    val_summary = tf.summary.scalar('validation loss', cost)

    #cost = tf.nn.l2_loss(pred - y)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = opt.minimize(cost)
    gradient_step = opt.compute_gradients(pred, tf.trainable_variables())

    init_op = tf.global_variables_initializer()

    var_names = tf.global_variables()
    restore_vars = [v for v in var_names if 'layer' not in v.name and 'ConvPred' not in v.name and 'Adam' not in v.name and 'power' not in v.name]
    #for i in range(len(restore_vars)):
    #    print restore_vars[i].name
    saver2 = tf.train.Saver(restore_vars)
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)#0.87777777)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #print('4')
        merge = tf.summary.merge_all()
        folderName = "./logs/Run - " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        os.makedirs(folderName, exist_ok=True)
        summary_writer = tf.summary.FileWriter(folderName, sess.graph)
       
        initsess = sess.run(init_op)
        print(initsess)
        # Load the converted parameters
        #print('Loading the model')
        #net.load(model, sess)
        #saver.save(sess, "resnet50_converted.ckpt" )
        #optinal

        #try:
        saver2.restore(sess, model)
        #except:
            #print('something wrong')


        # Training
        print('Training')
        
        official_split = scipy.io.loadmat(split_path)
        indices = np.squeeze(official_split['trainNdxs'], axis = 1)
        np.random.seed(0)
        np.random.shuffle(indices)
        shuffled_indices, validation_indices = np.split(indices,[int(0.7 * len(indices))])
        print(shuffled_indices)
        print(validation_indices)
        #shuffled_indices = np.arange(num_train_images)
        #validation_indices = np.arange(num_train_images+1 , num_train_images + num_valid_images)
        
        imgs, g_trths = augment_images(img_dir, shuffled_indices)
       
        # Keep training until reach max iterations
        for epoch in range(epochs):
            print('------------------------ Epoch ' + str(epoch) + ' ---------------------------')

            #np.random.shuffle(shuffled_indices)
            step = 1
            train_loss = 0.

            #train_loss = train_loss.astype(double)
            while step * batch_size < num_train_images:
               # batch_x, batch_y = next_batch_nyu2(batch_size, img_dir,
                # shuffled_indices[(step - 1) * batch_size : step * batch_size ],
                # out_height, out_width, 'train_shrink')
                batch_x = imgs[(step - 1) * batch_size : step * batch_size ]
                batch_y = g_trths[(step - 1) * batch_size : step * batch_size ]
                #print('Batch Shape: ' + str(batch_x.shape))
                #fig = plt.figure()
                #ii = plt.imshow(batch_x[0,:,:,0])
                #fig.colorbar(ii)
                #plt.show()
        
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, is_training: True})
                if(step == 1):
                    imageBatch = np.reshape(batch_x, (-1, height, width, channels))
                    tf.summary.image("Input images", imageBatch,  max_outputs= batch_size)
                
                #print "gradient size: " + str(len(gradient))
                #for gr in gradient:
                #    print len(gr)
                #loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, bg: batch_bg, keep_prob: 1.})
                #tf.reset_default_graph()
                
                #train_loss += loss
                # Calculate batch loss and accuracy
                loss,trloss_summ = sess.run([cost, train_summary], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
                
                #print('Loss for batch no '+ str(step) + ' ' + str(loss) )
                train_loss += loss


                if step % display_step == 0:
                    batch_x, batch_y = next_batch_nyu2(batch_size, img_dir,
                     shuffled_indices[0 : batch_size ], out_height, out_width, 'train')
                    loss = sess.run( cost, feed_dict={x: batch_x, y: batch_y,  keep_prob: 1., is_training: False})
                    print("Iter" + str(step*batch_size))
                    print(loss)
                    #summary_writer.add_summary(summary, step)

                step += 1
            summary_writer.add_summary(trloss_summ, epoch)
            print('TrainLoss after epoch ' + str(epoch) + ' : '  + str(train_loss/(step - 1)))


            '''
            # TRAINING-SET LOSS --------------------------------------------------------------------------
            step = 1
            train_loss = 0.

            while step * batch_size < num_train_images:
                batch_x, batch_y = next_batch(batch_size, img_dir,
                  shuffled_indices[(step - 1) * batch_size : step * batch_size ],
                  out_height, out_width, 'train')
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
                train_loss += loss
                step += 1
            print 'TrainLoss after epoch ' + str(epoch) + ' : '  + str(train_loss/(step - 1))
            '''

            # VALIDATION-SET LOSS --------------------------------------------------------------------------
            step = 1
            valid_loss = 0.

            while step * batch_size < num_valid_images:
                batch_x, batch_y = next_batch_nyu2(batch_size, img_dir,
                  validation_indices[(step - 1) * batch_size :step * batch_size ],
                  out_height, out_width, 'valid')
                loss,valloss_summ = sess.run([cost, val_summary], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
                
                '''
                prediction = sess.run(net.get_output(), feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: 0})
                plt.figure(0)
                plt.imshow(prediction[0, :, :, 0])
                plt.figure(1)
                plt.imshow(batch_x[0, :, :, :])
                plt.figure(2)
                plt.imshow(batch_y[0, :, :, 0])
                plt.show()
                '''
                valid_loss += loss
                step += 1
            print('ValidLoss after epoch ' + str(epoch) + ' : '  + str(valid_loss/(step - 1)))
            summary_writer.add_summary(valloss_summ, epoch)
            # --------------------------------------------------------------------------------------------


            save_path = saver.save(sess, "model_" + str(epoch) + ".ckpt" )
            #print 'Epoch: ' + epoch + ' , Saved' + save_path

        print("Optimization Finished!")

        summary_writer.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to model')
    parser.add_argument('images_path', help='Path to images')
    parser.add_argument('split_path', help='Path to official split file')
    args = parser.parse_args()

    # train the network
    train(args.model_path, args.images_path, args.split_path)


if __name__ == '__main__':
    main()
