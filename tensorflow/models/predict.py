import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageOps
from fcrn import *
import helpers as Intrinsic
import utils as utils
import h5py

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 5# 3
    batch_size = 1
    
    #For NYU
    #fx, fy ,cx, cy = Intrinsic.initIntrinsic()
    #final_fx, final_fy ,final_cx, final_cy = Intrinsic.resizeIntrinsic(228, 304 ,fx, fy ,cx, cy, 2.0) 
    #orig_metric_cord = Intrinsic.findMetricCordinates(final_fx, final_fy ,final_cx, final_cy, 304, 228)
    
    #Load Sun 3D dataset
    sun_imgs, sun_gts, dataset_label, indices_size = utils.sun3Ddataset()
    orig_metric_cord, crop_metric_cord, res_size = Intrinsic.findSun3DMetricCoords()
    index = 105
    
    #f = h5py.File(image_path)
        #official_split = scipy.io.loadmat(split_path)
        #test_indices = np.squeeze(official_split['testNdxs'], axis = 1)
    #img_file = f['images'][295]        
    #img_reshaped = np.transpose(img_file, (2, 1, 0))
    #img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
    
    img = Image.open(sun_imgs[index])
    
    border = (8, 6, 8, 6) # left, up, right, bottom
    cropped_img = ImageOps.crop(img, border)
    resized_img = cropped_img.resize((304, 228), Image.BILINEAR)
    #transfrmd_img = np.concatenate((np.asarray(resized_img), orig_metric_cord), axis = 2)
    transfrmd_img = np.concatenate((np.asarray(resized_img), orig_metric_cord[dataset_label[index]]), axis = 2)
    img = np.asarray(transfrmd_img).reshape(batch_size, height,width,channels)
    #img = np.asarray(resized_img).reshape(batch_size, height,width,channels)
    
    #Display input image
    inpfig = plt.figure()
    inp_img = plt.imshow(resized_img)
    plt.show()
    
    #Display groundtruth
    #depth = f['depths'][295].T
    dpth = Image.open(sun_gts[index])  
    #dpth = Image.fromarray(depth.astype(np.float64))
    cropped_dpth = ImageOps.crop(dpth, border)
    resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)
    #resized_dpth = np.expand_dims(resized_dpth, axis = 3)
    #dpth = np.array(dpth)/1000.
    

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
        pred = sess.run(net.get_output(), feed_dict={input_node: img}) #img
        
        # Plot result
        min_dpth = np.min(np.asarray(resized_dpth))
        max_dpth = np.max(np.asarray(resized_dpth))
        min_pred = np.min(pred[0,:,:,0])
        max_pred = np.max(pred[0,:,:,0])
        min_val = np.minimum(min_dpth,min_pred )
        max_val = np.maximum(max_dpth,max_pred )
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        #ax.set_xlim(xmin=0.0, xmax=1000)
        ax1.imshow(np.asarray(resized_dpth), interpolation='nearest')
        #plt.show()
        
        #fig = plt.figure()
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(pred[0,:,:,0], interpolation='nearest')
        #fig.colorbar(ii)
        #plt.axis('off')
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

        



