import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
import h5py
import imgaug.augmenters as iaa
import imgaug as ia
import helpers as Intrinsic
import scipy.io
import os, os.path

def relative_error(ground, prediction):

    valid = ground > 0.
    mask_low = prediction > 0.
    npix = np.sum(valid * mask_low)

    diff = abs(valid * prediction - mask_low * ground)
    rel_diff = np.where(ground > 0., diff / ground, 0.)
    rel = np.sum(rel_diff)
    rel/= npix

    return rel

def next_batch_nyu2(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []

    f = h5py.File(im_dir)
    for index in indices:

        img_file = f['images'][index-1]
        dpth_file = f['depths'][index-1].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)  
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)

        images.append(np.asarray(resized_img))
        depths.append(np.asarray(resized_dpth))
        
    images = np.asarray(images)
    depths = np.asarray(depths)

    return images, depths

def augment_images(im_dir, indices): 
    f = h5py.File(im_dir)
    augmented_imgs = []
    augmented_gts = []
    ia.seed(1)
    print(indices.shape)
    for index in indices:
        img_file = f['images'][index-1]
        dpth_file = f['depths'][index-1].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)       
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)      
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)
        
        resized_dpth_arr = np.asarray(resized_dpth)
        resized_dpth_arr[resized_dpth_arr< 0.5] = 0.0
        
        augmented_imgs.append(np.asarray(resized_img))
        augmented_gts.append(resized_dpth_arr)
        
        crops = 3
        #augment_operations(crops, cropped_img, cropped_dpth, resized_img, resized_dpth_arr, augmented_imgs, augmented_gts)
        
    augmented_imgs = np.asarray(augmented_imgs)
    augmented_gts = np.asarray(augmented_gts)
    
    return augmented_imgs,augmented_gts

def augment_operations(crops, cropped_img, cropped_dpth, resized_img, resized_dpth, augmented_imgs, augmented_gts):
    
    for i in range(crops):
        crop_img,col_cj,row_ci = Intrinsic.random_crop(np.asarray(cropped_img), (224, 304))
        resize_img = Image.fromarray(crop_img.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
        crop_dpth = np.asarray(cropped_dpth)[row_ci:row_ci+224, col_cj: col_cj + 304]
        resize_dpth = Image.fromarray(crop_dpth).resize((160, 128), Image.NEAREST)
        resize_dpth = np.expand_dims(resize_dpth, axis = 3)
        augmented_imgs.append(np.asarray(resize_img))
        augmented_gts.append(np.asarray(resize_dpth))
        
    angle = np.random.randint(-10,10)
    rotate = iaa.Affine(rotate=angle)
    rotated_img = rotate.augment_image(np.asarray(cropped_img))
    rotated_img = Image.fromarray(rotated_img.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
    rotated_dpth = rotate.augment_image(np.asarray(cropped_dpth))
    rotated_dpth = Image.fromarray(rotated_dpth).resize((160, 128), Image.NEAREST)
    rotated_dpth = np.expand_dims(rotated_dpth, axis = 3)
    augmented_imgs.append(np.asarray(rotated_img))
    augmented_gts.append(np.asarray(rotated_dpth))
        
    colorjitter = iaa.AddToHueAndSaturation((-60, 60))
    enhanced_img = colorjitter.augment_image(np.asarray(resized_img))
    augmented_imgs.append(np.asarray(enhanced_img))
    augmented_gts.append(np.asarray(resized_dpth))
    
    swapped_img = np.array(resized_img)
    red = swapped_img[:,:,0].copy()
    green = swapped_img[:,:,1].copy()
    swapped_img[:,:,1] = red
    swapped_img[:,:,0] = green
    augmented_imgs.append(np.asarray(swapped_img))
    augmented_gts.append(np.asarray(resized_dpth))
    
    brghtperturbator =  iaa.Multiply((0.8, 1.2), per_channel=0.2) #50-150% of original value
    perturbed_img = brghtperturbator.augment_image(np.asarray(resized_img))
    augmented_imgs.append(np.asarray(perturbed_img))
    augmented_gts.append(np.asarray(resized_dpth))
        
    flp2 = iaa.HorizontalFlip(1)
    flipped_img2 = flp2.augment_image(np.asarray(resized_img))
    flipped_dpth2 = flp2.augment_image(np.asarray(resized_dpth))
    augmented_imgs.append(np.asarray(flipped_img2))
    augmented_gts.append(np.asarray(flipped_dpth2))

def metric_images(im_dir, indices):
    
    fx, fy ,cx, cy = Intrinsic.initIntrinsic()
    final_fx, final_fy ,final_cx, final_cy = Intrinsic.resizeIntrinsic(228, 304 ,fx, fy ,cx, cy, 2.0) 
    orig_metric_cord = Intrinsic.findMetricCordinates(final_fx, final_fy ,final_cx, final_cy, 304, 228)
    
    # 3 crops and resize all crops to 304x228
    crop1_fx, crop1_fy ,crop1_cx, crop1_cy = Intrinsic.crop_and_resizeIntrinsic(423, 567, 468, 624,fx, fy ,cx, cy, 1.85) # 90% crop
    crop2_fx, crop2_fy ,crop2_cx, crop2_cy = Intrinsic.crop_and_resizeIntrinsic(375, 500, 468, 624,fx, fy ,cx, cy, 1.64) # 80% crop
    crop3_fx, crop3_fy ,crop3_cx, crop3_cy = Intrinsic.crop_and_resizeIntrinsic(354, 472, 468, 624,fx, fy ,cx, cy, 1.55) # 75% crop 
    
    crop_metric_cord1 = Intrinsic.findMetricCordinates(crop1_fx, crop1_fy ,crop1_cx, crop1_cy, 304, 228)
    crop_metric_cord2 = Intrinsic.findMetricCordinates(crop2_fx, crop2_fy ,crop2_cx, crop2_cy, 304, 228)
    crop_metric_cord3 = Intrinsic.findMetricCordinates(crop3_fx, crop3_fy ,crop3_cx, crop3_cy, 304, 228)
        
    f = h5py.File(im_dir)
    augmented_imgs = []
    augmented_gts = []
    ia.seed(1)
    #print(indices.shape)
    for index in indices:
        img_file = f['images'][index-1]
        dpth_file = f['depths'][index-1].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)       
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)      
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)
        resized_dpth_arr = np.array(resized_dpth)
        resized_dpth_arr[resized_dpth_arr< 0.5] = 0.0
        
        #print('Img shape ' + str(np.shape(np.asarray(resized_img))) )
        #print('Metric shape ' + str(np.shape(metric_cord)) )
        
        transfrmd_img = np.concatenate((np.asarray(resized_img), orig_metric_cord), axis = 2)
                
        #crops = 3
        #for i in range(crops):
        crop_img1,col_cj,row_ci = Intrinsic.random_crop(np.asarray(cropped_img), (423, 564))
        crop_img2,col_cj2,row_ci2 = Intrinsic.random_crop(np.asarray(cropped_img), (375, 500))
        crop_img3,col_cj3,row_ci3 = Intrinsic.random_crop(np.asarray(cropped_img), (354, 472))
        
        resize_img1 = Image.fromarray(crop_img1.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
        resize_img2 = Image.fromarray(crop_img2.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
        resize_img3 = Image.fromarray(crop_img3.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
        
        crop_transfrmd_img1 = np.concatenate((np.asarray(resize_img1), crop_metric_cord1), axis = 2)
        crop_transfrmd_img2 = np.concatenate((np.asarray(resize_img2), crop_metric_cord2), axis = 2)
        crop_transfrmd_img3 = np.concatenate((np.asarray(resize_img3), crop_metric_cord3), axis = 2)
        
        crop_dpth1 = np.asarray(cropped_dpth)[row_ci:row_ci+423, col_cj: col_cj + 564]
        crop_dpth2 = np.asarray(cropped_dpth)[row_ci2:row_ci2+375, col_cj2: col_cj2 + 500]
        crop_dpth3 = np.asarray(cropped_dpth)[row_ci3:row_ci3+354, col_cj3: col_cj3 + 472]
        
        resize_dpth1 = Image.fromarray(crop_dpth1).resize((160, 128), Image.NEAREST)
        resize_dpth2 = Image.fromarray(crop_dpth2).resize((160, 128), Image.NEAREST)
        resize_dpth3 = Image.fromarray(crop_dpth3).resize((160, 128), Image.NEAREST)
        
        resize_dpth1 = np.expand_dims(resize_dpth1, axis = 3)
        resize_dpth2 = np.expand_dims(resize_dpth2, axis = 3)
        resize_dpth3 = np.expand_dims(resize_dpth3, axis = 3)
        
        augmented_imgs.append(np.asarray(crop_transfrmd_img1))
        augmented_gts.append(np.asarray(resize_dpth1))
        augmented_imgs.append(np.asarray(crop_transfrmd_img2))
        augmented_gts.append(np.asarray(resize_dpth2))
        augmented_imgs.append(np.asarray(crop_transfrmd_img3))
        augmented_gts.append(np.asarray(resize_dpth3))
        
        augmented_imgs.append(np.asarray(transfrmd_img))
        augmented_gts.append(np.asarray(resized_dpth_arr))
        
        #crops = 3
        #augment_operations(crops, cropped_img, cropped_dpth, resized_img, resized_dpth, augmented_imgs, augmented_gts)
    #print('Shape ----- ' + str(np.shape(augmented_imgs)))   
    augmented_imgs = np.array(augmented_imgs, dtype=np.uint8)
    augmented_gts = np.asarray(augmented_gts)
    
    return augmented_imgs,augmented_gts

#Retrieve file locations of Sun 3D test dataset
def sun3Ddataset():
    
    path_to_sun3D = 'C:/Users/jeshw/Desktop/AppProject/FCRN-DepthPrediction/Sun3D_data/SUNRGBDtoolbox/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
    sun_indices = scipy.io.loadmat(path_to_sun3D)['alltest']
    sunindices_length =  np.array(sun_indices)[0].size
    #print(sunindices_length)
    testimages = []
    dataset = []
    testgts = []

    for index in np.array(sun_indices)[0]:
        sample = index[0]
        path = sample.split("SUNRGBD",1)[1]
        dataset_label = path.split("/")[2]
        image_path = 'C:/Users/jeshw/Desktop/AppProject/FCRN-DepthPrediction/Sun3D_data/SUNRGBD/SUNRGBD' + path + '/image'
        depth_path = 'C:/Users/jeshw/Desktop/AppProject/FCRN-DepthPrediction/Sun3D_data/SUNRGBD/SUNRGBD' + path + '/depth'
        #print(testdepth_path)
        image_file = image_path + '/' + os.listdir(image_path)[0]
        depth_file = depth_path + '/' + os.listdir(depth_path)[0]
        dataset.append(dataset_label)
        testimages.append(image_file)
        testgts.append(depth_file)
    
    return testimages, testgts, dataset, sunindices_length
