import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.io
from PIL import Image
from PIL import ImageOps
from fcrn import *
import helpers as Intrinsic
import utils as utils
import h5py

def evaluate(model_data_path, image_path, split_path):
  
    # Default input size
    height = 228
    width = 304
    channels = 3 #5
    #num_test_images = 654
    batch_size = 64
     
    #Load Sun 3D dataset
    sun_imgs, sun_gts, dataset_label, indices_size = utils.sun3Ddataset()
    
    abs_rels = []
    rmses = []
    log_errs = []
    
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    step = 1
    #print(indices_size)
    
    #get pre calculated metric coordinates for all sub datasets
    orig_metric_cord, crop_metric_cord, res_size = Intrinsic.findSun3DMetricCoords()
    
    #Testing in batches
    while step * batch_size <= indices_size: #indices.size:
        
        inp_batch = []
        gt_batch = []
        
        #res_size = [[423,564], [375, 500], [354, 472]] # 90%, 80%, 75% crop
        crops = len(res_size['NYUdata'])
        #print('Value ' + str(int(batch_size/(crops+1))))
        
        # Get the image and ground truth numpy arrays for entire batch 
        for index in range((step - 1) * int(batch_size/(crops+1)), step * int(batch_size/(crops+1))):
            print('Index: ' + str((step - 1) * int(batch_size/(crops+1))) + ' --->   ' + str( step * int(batch_size/(crops+1))))
            img = Image.open(sun_imgs[index])
            border = (8, 6, 8, 6) # left, up, right, bottom
            cropped_img = ImageOps.crop(img, border)
            resized_img = cropped_img.resize((304, 228), Image.BILINEAR)
            #transfrmd_img = np.concatenate((np.asarray(resized_img), orig_metric_cord[dataset_label[index]]), axis = 2)   
            
            inp_batch.append(np.asarray(resized_img))
            #inp_batch.append(transfrmd_img)
            
            dpth = Image.open(sun_gts[index])
            cropped_dpth = ImageOps.crop(dpth, border)
            resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)
            resized_dpth = np.expand_dims(resized_dpth, axis = 3)
            gt_batch.append(resized_dpth)
            #gt_batch = np.array(gt_batch)
            
            crop_img = []
            col_cj= []
            row_ci = []
            resize_img = []
            crop_transfrmd_img = []
            crop_dpth = []
            resize_dpth = []
            
            #print('crops ' + str(range(crops)))
            for i in range(crops):
                cropinfo = Intrinsic.center_crop(np.asarray(cropped_img), res_size[dataset_label[index]][i]) #Random or center crop
                crop_img.append(cropinfo[0])
                col_cj.append(cropinfo[1])
                row_ci.append(cropinfo[2])
                resize_img.append( Image.fromarray(crop_img[i].astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR))
                #crop_transfrmd_img.append( np.concatenate((np.asarray(resize_img[i]), crop_metric_cord[dataset_label[index]][i]), axis = 2))
                crop_dpth.append(np.asarray(cropped_dpth)[ row_ci[i]:row_ci[i]+res_size[dataset_label[index]][i][0], col_cj[i]: col_cj[i] + res_size[dataset_label[index]][i][1] ])
                resize_dpth.append(Image.fromarray(crop_dpth[i]).resize((160, 128), Image.NEAREST))
                resize_dpth_temp = np.array(resize_dpth[i])/10000.
                resize_dpth[i] = np.expand_dims(resize_dpth_temp, axis = 3)
                
                inp_batch.append(np.asarray(resize_img[i])) #crop_transfrmd_img[i]
                gt_batch.append(np.asarray(resize_dpth[i]))
            
        print('-------Batch: ' + str(step) + ' Images: ' + str(step * batch_size) + '-----------------')
        #print('Shape ' + str(np.squeeze(np.array(inp_batch), axis=1).shape))
        print('Shape ' + str(np.array(inp_batch).shape))
        #print('Shape ' + str(np.array(gt_batch).shape))
        
        with tf.Session() as sess:

            # load from trained ckpt file
            saver = tf.train.Saver()     
            saver.restore(sess, model_data_path)

            # Predict depth for entire batch
            #pred = sess.run(net.get_output(), feed_dict={input_node: np.squeeze(np.array(inp_batch), axis=1)})
            pred = sess.run(net.get_output(), feed_dict={input_node: np.array(inp_batch)})
            print( 'The predicted depth map shape: ' + str(pred.shape))
            
        #Calculate error values for all predictions in the batch
        for ind in range(batch_size):
            
            mask = np.array(gt_batch)[ind] > 0.5 #assuming you have meters
            numpix = np.sum(mask)
            gt_masked = mask * np.array(gt_batch)[ind]
            print( 'The ground truth shape: ' + str(np.array(gt_batch)[ind].shape))
            pred_masked = mask * np.array(pred)[ind]
            epsilon = 0.00001
            #print('Min value ' + str(np.min(np.array(pred)[ind])))
            
            #abs_rel = np.mean(np.abs(np.array(gt_batch)[ind] - pred[ind]) / np.array(gt_batch)[ind])
            abs_rel = np.sum(np.abs(gt_masked - pred_masked) / (gt_masked + epsilon)) / numpix
            rmse = (np.array(gt_batch)[ind] - pred[ind]) ** 2
            rmse = np.sqrt(rmse.mean())
            log_10 = (np.abs(np.log10(np.array(gt_batch)[ind]+0.0000000001)-np.log10(pred[ind]+0.0000000001))).mean()
            
            #Appending to Error arrays
            abs_rels.append(abs_rel)
            rmses.append(rmse)
            log_errs.append(log_10)
            
        step+=1
    
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

    # Evaluate the testset
    evaluate(args.model_path, args.image_paths, args.split_path)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



