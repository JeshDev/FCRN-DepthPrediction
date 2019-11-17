import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageOps
from fcrn import *
import h5py

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    #img = Image.open(image_path)
    #img = img.resize([width,height], Image.ANTIALIAS)
    #img = np.array(img).astype('float32')
    #img = np.expand_dims(np.asarray(img), axis = 0)
    
    f = h5py.File(image_path)
    #official_split = scipy.io.loadmat(split_path)
    #test_indices = np.squeeze(official_split['testNdxs'], axis = 1)
    img_file = f['images'][1433]
    img_reshaped = np.transpose(img_file, (2, 1, 0))
    img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
    border = (16, 16, 16, 16) # left, up, right, bottom
    cropped_img = ImageOps.crop(img, border)
    resized_img = cropped_img.resize((304, 228), Image.BILINEAR)
    img = np.asarray(resized_img).reshape(batch_size, height,width,channels)
    
    #Display input image
    inpfig = plt.figure()
    inp_img = plt.imshow(resized_img)
    plt.show()
    
    #Display groundtruth
    depth = f['depths'][1433].T
    dpth = Image.fromarray(depth.astype(np.float64))
    resized_dpth = dpth.resize((160, 128), Image.NEAREST)
    dpth = np.array(dpth)/1000.
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(np.asarray(resized_dpth))
    #plt.show()

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        #fig = plt.figure()
        #print('Output shape: ' + str(pred.shape))
        #ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        ax2 = fig.add_subplot(1,2,2)
        ii = ax2.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.axis('off')
        plt.show()
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



