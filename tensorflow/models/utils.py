import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
import h5py
import imgaug.augmenters as iaa
import imgaug as ia
import helpers as Intrinsic

def pixel_accuracy(ground, prediction):
    match = (ground == prediction)

    return 1. * np.count_nonzero(match)/ (ground.shape[0] * ground.shape[1])

def iou(ground, prediction):
    #print "la"
    intersection = ground * prediction
    union = ground + prediction
    iou_ones =  1. * np.count_nonzero(intersection) / np.count_nonzero(union)

    intersection = ground + prediction
    union = ground * prediction

    iou_zeros =  1. * np.count_nonzero(intersection==0.) / (np.count_nonzero(union==0.)+1)

    return iou_ones, iou_zeros


def relative_error(ground, prediction):

    valid = ground > 0.
    mask_low = prediction > 0.
    npix = np.sum(valid * mask_low)

    diff = abs(valid * prediction - mask_low * ground)
    rel_diff = np.where(ground > 0., diff / ground, 0.)
    rel = np.sum(rel_diff)
    rel/= npix

    return rel

def next_batch_seg(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    segmentation = []


    #img_dir = im_dir + '/' + subdir
    img_dir = im_dir

    for index in indices:

        img = Image.open(img_dir + '/color' + str(index).rjust(5, '0') + '.jpg')
        seg = Image.open(img_dir + '/segmentation' + str(index).rjust(5, '0') + '.png')

        area = (10, 0, 310, 240)
        img = img.crop(area)
        seg = seg.crop(area)
        img = img.resize((320, 256), Image.BILINEAR)
        seg = seg.resize((160, 128), Image.NEAREST)
        seg = np.array(seg).astype(np.float)
        images.append(np.array(img))
        segmentation.append(seg)
        #print seg

    images = np.asarray(images)
    segmentation = np.asarray(segmentation)
    segmentation = np.expand_dims(segmentation, axis = 3)

    return images, segmentation

def next_batch_real(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []
    bg = []

    #img_dir = im_dir + '/' + subdir
    img_dir = im_dir

    for index in indices:
        #index = i + batch_index % num_images + 1

        img = Image.open(img_dir + '/rgb' + str(index) + '_new.png')

        dpt1 = Image.open(img_dir + '/depth' + str(index)+ '_new.png')
        #dpt2 = Image.open(img_dir + '/bg' + str(index).rjust(5, '0') + '.png')
        img = img.resize((320, 240), Image.BILINEAR)
        dpt1 = dpt1.resize((2*160, 2*120), Image.NEAREST)

        area = (8, 6, 320 - 8, 240 - 6)
        img = img.crop(area)
        dpt1 = dpt1.crop(area)
        #dpt2 = dpt2.crop(area)
        #image = image[:,10:310]
        dpt1 = dpt1.resize((2*160, 2*128), Image.NEAREST)
        img = img.resize((320, 256), Image.BILINEAR)
        #dpt2 = dpt2.resize((160, 128), Image.NEAREST)
        #dpt = dpt.resize((out_width, out_height), PIL.Image.NEAREST)
        dpt1 = np.array(dpt1).astype(np.float)/1000.
        #dpt2 = np.array(dpt2).astype(np.float)/1000.

        images.append(np.array(img))


        '''
        plt.figure(0)
        plt.imshow(np.array(img))
        plt.figure(1)
        plt.imshow(fground)
        plt.show()

        ldi = dpt2 > 0
        single = dpt2 <= 0
        dpt2 = ldi * dpt2 + single * dpt1
        '''
        #dpt = np.stack([dpt1, dpt2], axis = 0)
        depths.append(dpt1)

        #bg.append(dpt2)
    images = np.asarray(images)
    depths = np.asarray(depths)
    #bg = np.asarray(bg)
    depths = np.expand_dims(depths, axis = 3)
    #bg = np.expand_dims(bg, axis = 3)

    return images, depths

def next_batch_nyu(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []
    bg = []

    #img_dir = im_dir + '/' + subdir
    img_dir = im_dir

    for index in indices:
        #index = i + batch_index % num_images + 1

        img = Image.open(img_dir + '/rgb/' + str(index) + '.jpg')

        dpt1 = Image.open(img_dir + '/depth/' + str(index)+ '.png')
        #dpt2 = Image.open(img_dir + '/bg' + str(index).rjust(5, '0') + '.png')
        img = img.resize((320, 240), Image.BILINEAR)
        dpt1 = dpt1.resize((2*160, 2*120), Image.NEAREST)

        area = (8, 6, 320 - 8, 240 - 6)
        img = img.crop(area)
        dpt1 = dpt1.crop(area)
        #dpt2 = dpt2.crop(area)
        #image = image[:,10:310]
        dpt1 = dpt1.resize((2*160, 2*128), Image.NEAREST)
        img = img.resize((320, 256), Image.BILINEAR)
        #dpt2 = dpt2.resize((160, 128), Image.NEAREST)
        #dpt = dpt.resize((out_width, out_height), PIL.Image.NEAREST)
        dpt1 = np.array(dpt1).astype(np.float)/1000.
        #dpt2 = np.array(dpt2).astype(np.float)/1000.

        images.append(np.array(img))


        '''
        plt.figure(0)
        plt.imshow(np.array(img))
        plt.figure(1)
        plt.imshow(fground)
        plt.show()

        ldi = dpt2 > 0
        single = dpt2 <= 0
        dpt2 = ldi * dpt2 + single * dpt1
        '''
        #dpt = np.stack([dpt1, dpt2], axis = 0)
        depths.append(dpt1)

        #bg.append(dpt2)
    images = np.asarray(images)
    depths = np.asarray(depths)
    #bg = np.asarray(bg)
    depths = np.expand_dims(depths, axis = 3)
    #bg = np.expand_dims(bg, axis = 3)

    return images, depths

def next_batch(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []
    bg = []

    #img_dir = im_dir + '/' + subdir
    img_dir = im_dir

    for index in indices:
        #index = i + batch_index % num_images + 1

        img = Image.open(img_dir + '/color' + str(index).rjust(5, '0') + '.jpg')

        dpt1 = Image.open(img_dir + '/fg' + str(index).rjust(5, '0') + '.png')
        #dpt2 = Image.open(img_dir + '/bg' + str(index).rjust(5, '0') + '.png')

        area = (10, 0, 310, 240)
        img = img.crop(area)
        dpt1 = dpt1.crop(area)
        #dpt2 = dpt2.crop(area)
        #image = image[:,10:310]
        img = img.resize((320, 256), Image.BILINEAR)
        dpt1 = dpt1.resize((2*160, 2*128), Image.NEAREST)
        #dpt2 = dpt2.resize((160, 128), Image.NEAREST)
        #dpt = dpt.resize((out_width, out_height), PIL.Image.NEAREST)
        dpt1 = np.array(dpt1).astype(np.float)/1000.
        #dpt2 = np.array(dpt2).astype(np.float)/1000.

        images.append(np.array(img))


        '''
        plt.figure(0)
        plt.imshow(np.array(img))
        plt.figure(1)
        plt.imshow(fground)
        plt.show()

        ldi = dpt2 > 0
        single = dpt2 <= 0
        dpt2 = ldi * dpt2 + single * dpt1
        '''
        #dpt = np.stack([dpt1, dpt2], axis = 0)
        depths.append(dpt1)
        #bg.append(dpt2)
    images = np.asarray(images)
    depths = np.asarray(depths)
    #bg = np.asarray(bg)
    depths = np.expand_dims(depths, axis = 3)
    #bg = np.expand_dims(bg, axis = 3)

    return images, depths

def augment_images(im_dir, indices): 
    f = h5py.File(im_dir)
    augmented_imgs = []
    augmented_gts = []
    ia.seed(1)
    print(indices.shape)
    for index in indices:
        img_file = f['images'][index]
        dpth_file = f['depths'][index].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)       
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)      
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)
        
        augmented_imgs.append(np.asarray(resized_img))
        augmented_gts.append(np.asarray(resized_dpth))
        
        crops = 3
        augment_operations(crops, cropped_img, cropped_dpth, resized_img, resized_dpth, augmented_imgs, augmented_gts)
        
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
    
    new_fx, new_fy ,new_cx, new_cy = Intrinsic.initIntrinsic()
    metric_cord = Intrinsic.findMetricCordinates(new_fx, new_fy ,new_cx, new_cy, 304, 228)
    final_fx, final_fy ,final_cx, final_cy = Intrinsic.crop_and_resizeIntrinsic(114, 152, 228, 304,new_fx, new_fy ,new_cx, new_cy, 2.0)
    crop_metric_cord = Intrinsic.findMetricCordinates(final_fx, final_fy ,final_cx, final_cy, 304, 228)
        
    f = h5py.File(im_dir)
    augmented_imgs = []
    augmented_gts = []
    ia.seed(1)
    print(indices.shape)
    for index in indices:
        img_file = f['images'][index]
        dpth_file = f['depths'][index].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)       
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)      
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)
        
        transfrmd_img = np.concatenate((np.asarray(resized_img), metric_cord), axis = 2)
                
        crops = 2
        for i in range(crops):
            crop_img,col_cj,row_ci = Intrinsic.random_crop(np.asarray(cropped_img), (114, 152))
            resize_img = Image.fromarray(crop_img.astype(np.uint8), 'RGB').resize((304, 228), Image.BILINEAR)
            crop_transfrmd_img = np.concatenate((np.asarray(resize_img), crop_metric_cord), axis = 2)
            crop_dpth = np.asarray(cropped_dpth)[row_ci:row_ci+224, col_cj: col_cj + 304]
            resize_dpth = Image.fromarray(crop_dpth).resize((160, 128), Image.NEAREST)
            resize_dpth = np.expand_dims(resize_dpth, axis = 3)
            augmented_imgs.append(np.asarray(crop_transfrmd_img))
            augmented_gts.append(np.asarray(resize_dpth))
        
        augmented_imgs.append(np.asarray(transfrmd_img))
        augmented_gts.append(np.asarray(resized_dpth))
        
        #crops = 3
        #augment_operations(crops, cropped_img, cropped_dpth, resized_img, resized_dpth, augmented_imgs, augmented_gts)
    print('Shape ----- ' + str(np.shape(augmented_imgs)))   
    augmented_imgs = np.asarray(augmented_imgs)
    augmented_gts = np.asarray(augmented_gts)
    
    return augmented_imgs,augmented_gts

def next_batch_nyu2(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []
    
    #path_to_depth =  'NYU_data/nyu_depth_v2_labeled.mat'
    f = h5py.File(im_dir)
    for index in indices:

        img_file = f['images'][index]
        dpth_file = f['depths'][index].T
        
        img_reshaped = np.transpose(img_file, (2, 1, 0))
        
        img= Image.fromarray(img_reshaped.astype(np.uint8), 'RGB')
        dpth = Image.fromarray(dpth_file.astype(np.float64))
        
        border = (8, 6, 8, 6) # left, up, right, bottom
        cropped_img = ImageOps.crop(img, border)
        cropped_dpth = ImageOps.crop(dpth, border)
        
        resized_img = cropped_img.resize((304, 228), Image.BILINEAR)
        
        resized_dpth = cropped_dpth.resize((160, 128), Image.NEAREST)
        
        resized_dpth = np.expand_dims(resized_dpth, axis = 3)
        
        #dpth -= dpth.min()
        #dpth /= (dpth.max()-dpth.min())
        #dpth = np.array(resized_dpth)/1000.
        images.append(np.asarray(resized_img))
        depths.append(np.asarray(resized_dpth))
        
    images = np.asarray(images)
    #metric_cord = Intrinsic.findMetricCordinates()
    #transfrmd_img = np.append(np.asarray(resized_img), metric_cord, axis = 2)
    depths = np.asarray(depths)

    return images, depths

def next_batch_joint(size, im_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []
    segs = []

    #img_dir = im_dir + '/' + subdir
    img_dir = im_dir

    for index in indices:
        #index = i + batch_index % num_images + 1

        img = Image.open(img_dir + '/color' + str(index).rjust(5, '0') + '.jpg')

        dpt1 = Image.open(img_dir + '/fg' + str(index).rjust(5, '0') + '.png')
        seg = Image.open(img_dir + '/segmentation' + str(index).rjust(5, '0') + '.png')

        area = (10, 0, 310, 240)
        img = img.crop(area)
        dpt1 = dpt1.crop(area)
        seg = seg.crop(area)
        #image = image[:,10:310]
        img = img.resize((320, 256), Image.BILINEAR)
        dpt1 = dpt1.resize((2*160, 2*128), Image.NEAREST)
        seg = seg.resize((2*160, 2*128), Image.NEAREST)
        #dpt2 = dpt2.resize((160, 128), Image.NEAREST)
        #dpt = dpt.resize((out_width, out_height), PIL.Image.NEAREST)
        dpt1 = np.array(dpt1).astype(np.float)/1000.
        seg = np.array(seg).astype(np.float)

        images.append(np.array(img))


        '''
        plt.figure(0)
        plt.imshow(np.array(img))
        plt.figure(1)
        plt.imshow(fground)
        plt.show()

        ldi = dpt2 > 0
        single = dpt2 <= 0
        dpt2 = ldi * dpt2 + single * dpt1
        '''
        #dpt = np.stack([dpt1, dpt2], axis = 0)
        depths.append(dpt1)
        segs.append(seg)
    images = np.asarray(images)
    depths = np.asarray(depths)
    #bg = np.asarray(bg)
    depths = np.expand_dims(depths, axis = 3)
    segs = np.expand_dims(segs, axis = 3)

    return images, depths, segs

def next_scannet_batch(size, img_dir, indices, out_height, out_width, subdir ):
    images = []
    depths = []

    for index in indices:
        #index = i + batch_index % num_images + 1

        img = Image.open(img_dir + '/color/' + str(index) + '.jpg')

        dpt1 = Image.open(img_dir + '/depth/' + str(index) + '.png')
        #dpt2 = Image.open(img_dir + '/bg' + str(index).rjust(5, '0') + '.png')
        img = img.resize((640,480), Image.BILINEAR)

        area = (20, 0, 620, 480)
        img = img.crop(area)
        dpt1 = dpt1.crop(area)
        #dpt2 = dpt2.crop(area)
        #image = image[:,10:310]
        img = img.resize((640, 512), Image.BILINEAR)
        dpt1 = dpt1.resize((320, 256), Image.NEAREST)
        #dpt2 = dpt2.resize((160, 128), Image.NEAREST)
        #dpt = dpt.resize((out_width, out_height), PIL.Image.NEAREST)
        dpt1 = np.array(dpt1).astype(np.float)/1000.
        #dpt2 = np.array(dpt2).astype(np.float)/1000.

        images.append(np.array(img))

        #dpt = np.stack([dpt1, dpt2], axis = 0)
        depths.append(dpt1)
        #bg.append(dpt2)
    images = np.asarray(images)
    depths = np.asarray(depths)
    #bg = np.asarray(bg)
    depths = np.expand_dims(depths, axis = 3)
    #bg = np.expand_dims(bg, axis = 3)

    return images, depths
