import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.io
from PIL import Image
from PIL import ImageOps
from fcrn import *
import h5py

def evaluate(model_data_path, image_path, split_path):
  
    # Default input size
    height = 228
    width = 304
    channels = 3
    num_test_images = 654
    batch_size = 64
    
    #Load images
    f = h5py.File(image_path)
    official_split = scipy.io.loadmat(split_path) 
    indices = np.squeeze(official_split['testNdxs'], axis = 1) #indices from official NYU split
    
    abs_rels = []
    rmses = []
    log_errs = []
    
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    step = 1
    print(indices)
    
    #Testing in batches
    while step * batch_size <= indices.size:
        
        inp_batch = []
        gt_batch = []
        
        # Get the image and ground truth numpy arrays for entire batch 
        for index in indices[(step - 1) * batch_size : step * batch_size]:
        
            img_file = f['images'][index]
            img_reshaped = np.transpose(img_file, (2, 1, 0))
            img = Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
            border = (16, 16, 16, 16) # left, up, right, bottom
            cropped_img = ImageOps.crop(img, border)
            resized_img = cropped_img.resize((304, 228), Image.BILINEAR)
            img = np.asarray(resized_img).reshape(1, height, width, channels)
            inp_batch.append(img)
            #inp_batch = np.squeeze(np.array(inp_batch), axis=1)
            
            dpth_file = f['depths'][index].T
            dpth = Image.fromarray(dpth_file.astype(np.float64))        
            resized_dpth = dpth.resize((160, 128), Image.NEAREST)
            resized_dpth = np.expand_dims(resized_dpth, axis = 3)
            gt_batch.append(resized_dpth)
            #gt_batch = np.array(gt_batch)
            
        print('-------Batch: ' + str(step) + ' Images: ' + str(step * batch_size) + '-----------------')
        print('Shape ' + str(np.squeeze(np.array(inp_batch), axis=1).shape))
        print('Shape ' + str(np.array(gt_batch).shape))
        
        with tf.Session() as sess:

            # load from trained ckpt file
            saver = tf.train.Saver()     
            saver.restore(sess, model_data_path)

            # Predict depth for entire batch
            pred = sess.run(net.get_output(), feed_dict={input_node: np.squeeze(np.array(inp_batch), axis=1)})
            #print( 'The predicted depth map shape: ' + str(pred.shape))
            
            '''
            if step == 1:
                #Display sample input image
                inpfig = plt.figure()
                inp_img = plt.imshow(np.squeeze(np.array(inp_batch), axis=1)[10])
                plt.show()
    
                #Display sample groundtruth and predicted depth map
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                gt_test = gt_batch[10]
                ax1.imshow(np.asarray(gt_test[:,:,0]))
                ax2 = fig.add_subplot(1,2,2)
                pred_test = pred[10]
                ii = ax2.imshow(pred_test[:,:,0], interpolation='nearest')
                fig.colorbar(ii)
                plt.axis('off')
                plt.show()
            '''
        #Calculate error values for all predictions in the batch
        for ind in range(batch_size):
            
            mask = np.array(gt_batch)[ind] > 0.3 #assuming you have meters
            numpix = np.sum(mask)
            gt_masked = mask * np.array(gt_batch)[ind]
            pred_masked = mask * np.array(pred)[ind]
            epsilon = 0.00001
            
            #abs_rel = np.mean(np.abs(np.array(gt_batch)[ind] - pred[ind]) / np.array(gt_batch)[ind])
            abs_rel = np.sum(np.abs(gt_masked - pred_masked) / (gt_masked + epsilon)) / numpix
            rmse = (np.array(gt_batch)[ind] - pred[ind]) ** 2
            rmse = np.sqrt(rmse.mean())
            log_10 = (np.abs(np.log10(np.array(gt_batch)[ind])-np.log10(pred[ind]))).mean()
            
            #Appending to Error arrays
            abs_rels.append(abs_rel)
            rmses.append(rmse)
            log_errs.append(log_10)
            
        step+=1
        
        
        
    #print('Relative error: ' + str(np.array(abs_rels)))
    #print('RMS error: ' + str(np.array(rmses))) 
    #print('Log 10 error: ' + str(np.array(log_errs)))
    
    #Finding mean of errors from entire test set
    print('--------------------------------------------------------')
    print('Relative mean error: ' + str(np.mean(np.array(abs_rels))))
    print('RMS mean error: ' + str(np.mean(np.array(rmses)))) 
    print('Log 10 mean error: ' + str(np.mean(np.array(log_errs)))) 
                
def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    parser.add_argument('split_path', help='Path to official split file')
    args = parser.parse_args()

    # Predict the image
    evaluate(args.model_path, args.image_paths, args.split_path)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



