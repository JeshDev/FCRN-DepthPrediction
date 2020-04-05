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
    channels = 5
    batch_size = 4 #16
    out_height = 128
    out_width = 160
    out_channels = 1

    # Create a placeholder for the input image
    x = tf.placeholder(tf.float32, shape = (None, height, width, channels))
    y = tf.placeholder(tf.float32, shape = (None, out_height, out_width, out_channels))

    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # Construct the network
    net = ResNet50UpProj({'data': x}, batch_size, keep_prob, is_training)
    
    #Hyperparameters
    display_step = 500
    epochs = 2
    dropout = 0.5
    learning_rate = 0.001
    num_train_images = 795
    num_valid_images = 654
    
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
    #decay_rate = learning_rate / epochs
    #g_step = tf.get_variable('global_step', trainable=False, initializer=0)
    #learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, global_step = g_step, decay_steps=num_train_images, decay_rate=decay_rate, staircase=True)
    
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #opt = tf.train.AdamOptimizer(learning_rate=learning_rate_node)

    optimizer = opt.minimize(cost)
    #optimizer = opt.minimize(cost, global_step=g_step)
    
    gradient_step = opt.compute_gradients(pred, tf.trainable_variables())
    init_op = tf.global_variables_initializer()

    var_names = tf.global_variables()
    restore_vars = [v for v in var_names if 'layer' not in v.name and 'ConvPred' not in v.name and 'Adam' not in v.name and 'power' not in v.name and 'conv1' not in v.name] #and 'global_step' not in v.name]
    
    #for i in range(len(restore_vars)):
    #    print restore_vars[i].name
    
    saver2 = tf.train.Saver(restore_vars)
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)#0.87777777)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        #Log to Tensorboard
        merge = tf.summary.merge_all()
        folderName = "./logs/Run - " + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        os.makedirs(folderName, exist_ok=True)
        summary_writer = tf.summary.FileWriter(folderName, sess.graph)
       
        initsess = sess.run(init_op)
        print(initsess)

        #try:
        saver2.restore(sess, model)
        #except:
            #print('something wrong')


        # Training
        print('Training')
        
        #Get indices from official split file
        official_split = scipy.io.loadmat(split_path)
        train_indices = np.squeeze(official_split['trainNdxs'], axis = 1)
        validation_indices = np.squeeze(official_split['testNdxs'], axis = 1)
        #np.random.seed(0)
        #np.random.shuffle(train_indices)
        #shuffled_indices, validation_indices = np.split(train_indices,[int(0.7 * len(train_indices))])
        #print(shuffled_indices)
        print(train_indices)
        print(validation_indices)
        #shuffled_indices = np.arange(num_train_images)
        #validation_indices = np.arange(num_train_images+1 , num_train_images + num_valid_images)
        
        #imgs, g_trths = augment_images(img_dir, shuffled_indices)
        imgs, g_trths = metric_images(img_dir, train_indices)
       
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
        
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, is_training: True})
                if(step == 1):
                    imageBatch = np.reshape(batch_x, (-1, height, width, channels))
                    tf.summary.image("Input images", imageBatch,  max_outputs= batch_size)
                         
                # Calculate batch loss and accuracy
                loss,trloss_summ = sess.run([cost, train_summary], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
                
                #print('Loss for batch no '+ str(step) + ' ' + str(loss) )
                train_loss += loss

                if step % display_step == 0:
                    batch_x, batch_y = next_batch_nyu2(batch_size, img_dir,
                     train_indices[0 : batch_size ], out_height, out_width, 'train')
                    loss = sess.run( cost, feed_dict={x: batch_x, y: batch_y,  keep_prob: 1., is_training: False})
                    print("Iter" + str(step*batch_size))
                    print(loss)
                    #summary_writer.add_summary(summary, step)

                step += 1
            summary_writer.add_summary(trloss_summ, epoch)
            print('TrainLoss after epoch ' + str(epoch) + ' : '  + str(train_loss/(step - 1)))

            # VALIDATION-SET LOSS --------------------------------------------------------------------------
            step = 1
            valid_loss = 0.
            v_imgs, v_g_trths = metric_images(img_dir, validation_indices)
            while step * batch_size < num_valid_images:
                #batch_x, batch_y = next_batch_nyu2(batch_size, img_dir,
                  #validation_indices[(step - 1) * batch_size :step * batch_size ],
                  #out_height, out_width, 'valid')
                batch_x = v_imgs[(step - 1) * batch_size : step * batch_size ]
                batch_y = v_g_trths[(step - 1) * batch_size : step * batch_size ]
                loss,valloss_summ = sess.run([cost, val_summary], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})
                valid_loss += loss
                step += 1
            print('ValidLoss after epoch ' + str(epoch) + ' : '  + str(valid_loss/(step - 1)))
            summary_writer.add_summary(valloss_summ, epoch)
            # --------------------------------------------------------------------------------------------

            save_path = saver.save(sess, "model_" + str(epoch) + ".ckpt" )

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
