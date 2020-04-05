import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
import h5py
import operator


def findMetricCordinates(fx_rgb, fy_rgb, cx_rgb, cy_rgb, width_px, height_px):

    cam_matrix = np.array([[fx_rgb,0,cx_rgb],[0, fy_rgb, cy_rgb],[0,0,1]])
    inverse_cam_matrix = np.linalg.inv(cam_matrix)
    nx, ny = (width_px, height_px)
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    xv, yv = np.meshgrid(x, y)
    zv = np.ones((ny,nx))
    pixel_cord = np.stack((xv,yv,zv), axis=2)
    metric_cord = np.empty(shape=(ny,nx,3))
    for i in range(ny):
        for j in range(nx):
            metric_cord[i][j] = np.dot(inverse_cam_matrix, pixel_cord[i][j])
    return metric_cord[:,:,:2]

def initIntrinsic():
    
    #Kinect camera coordinates
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    crop_cx_rgb = cx_rgb - 8 #because of white border cropping
    crop_cy_rgb = cy_rgb - 6  
    
    cropped_width = 624
    cropped_height = 468
    scale = 0.48717
    #new_fx, new_fy ,new_cx, new_cy = resizeIntrinsic(cropped_height, cropped_width, fx_rgb, fy_rgb, crop_cx_rgb, crop_cy_rgb, scale) #resized to (228, 304, 3)
    
    #return new_fx, new_fy ,new_cx, new_cy
    return fx_rgb, fy_rgb ,crop_cx_rgb, crop_cy_rgb
    
def crop_and_resizeIntrinsic(crop_height, crop_width, orig_height, orig_width, fx_rgb, fy_rgb, cx_rgb, cy_rgb, scale):
    
    #new_cx_rgb = cx_rgb + float(width-1)/2 - crop_col_cj
    #new_cy_rgb = cy_rgb + float(height-1)/2 - crop_row_ci
    
    #Find new principal point of cropped image
    offset_cx = (orig_height - crop_height)/2
    offset_cy = (orig_width - crop_width)/2
    new_cx_rgb = cx_rgb - offset_cx
    new_cy_rgb = cy_rgb - offset_cy
    
    #Get modified prinicipal point and focal length after resize
    new_fx, new_fy ,new_cx, new_cy = resizeIntrinsic(crop_height, crop_width, fx_rgb, fy_rgb, new_cx_rgb, new_cy_rgb, scale)
    
    return new_fx, new_fy ,new_cx, new_cy

def resizeIntrinsic(height, width, fx, fy, cx_rgb, cy_rgb, scale):
    
    #Find original prinicipal point
    center_x = float(width-1) / 2
    center_y = float(height-1) / 2
    orig_cx_diff = cx_rgb - center_x
    orig_cy_diff = cy_rgb - center_y
    
    #Find center of scaled image
    scaled_height = round(scale * height)
    scaled_width = round(scale * width)
    scaled_center_x = float(scaled_width-1) / 2
    scaled_center_y = float(scaled_height-1) / 2
    
    #Find new focal length
    new_fx = scale * fx
    new_fy = scale * fy
    
    #Find new prinicipal point
    new_cx = scaled_center_x + scale * orig_cx_diff
    new_cy = scaled_center_y + scale * orig_cy_diff
    
    return new_fx, new_fy ,new_cx, new_cy

#Make Random crop of the image
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    print('shape--- ' + str(img.shape))
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], x, y

#To get a center crop of a image
def center_crop(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices], start, end

#Intrinsic parameters of different folders in Sun3D dataset
def sun3Dintrinsic(cam):
    
    if cam == 'b3dodata':
        fx_rgb = 520.532000
        fy_rgb = 520.744400
        cx_rgb = 277.925800
        cy_rgb = 215.115000
        height = 415 #427 - 12
        width = 545 #561 - 16
        crop_size = [[374,491], [332, 436], [311, 409]] # 90%, 80%, 75% crop
    elif cam == 'NYUdata':
        fx_rgb = 518.857901
        fy_rgb = 519.469611
        cx_rgb = 284.582449
        cy_rgb = 208.736166
        height = 415 #427 - 12
        width = 545 #561- 16
        crop_size = [[374,491], [332, 436], [311, 409]] # 90%, 80%, 75% crop
    elif cam == 'align_kv2' or cam == 'kinect2data':
        fx_rgb = 529.500000
        fy_rgb = 529.500000
        cx_rgb = 365.000000
        cy_rgb = 265.000000
        height = 518 #530 - 12
        width = 714 #730 - 16
        crop_size = [[436,643], [414, 571], [388, 535]] # 90%, 80%, 75% crop
    elif cam == 'lg' or cam == 'sa' or cam == 'shr':
        fx_rgb = 693.744690
        fy_rgb = 693.744690
        cx_rgb = 360.431915
        cy_rgb = 264.750000
        height = 519 #531 - 12
        width = 665 #681 - 16
        crop_size = [[467,598], [415, 532], [389, 499]] # 90%, 80%, 75% crop
    elif cam == 'sh':
        fx_rgb = 691.584229
        fy_rgb = 691.584229
        cx_rgb = 362.777557
        cy_rgb = 264.750000
        height = 519 #531 - 12
        width = 665 #681- 16
        crop_size = [[467,598], [415, 532], [389, 499]] # 90%, 80%, 75% crop
    elif cam == 'sun3ddata':
        fx_rgb = 570.342205
        fy_rgb = 570.342205
        cx_rgb = 310.000000
        cy_rgb = 225.000000
        height = 429 #441 - 12
        width = 575 #591- 16
        crop_size = [[386, 517], [343, 460], [322, 431]] # 90%, 80%, 75% crop
    elif cam == 'xtion_align_data':
        fx_rgb = 570.342224
        fy_rgb = 570.342224
        cx_rgb = 291.000000
        cy_rgb = 231.000000
        height = 429 #441 - 12
        width = 575 #591- 16
        crop_size = [[386, 517], [343, 460], [322, 431]] # 90%, 80%, 75% crop
        
    crop_cx_rgb = cx_rgb - 8 #because of white border cropping
    crop_cy_rgb = cy_rgb - 6
    
    return fx_rgb, fy_rgb ,crop_cx_rgb, crop_cy_rgb, height, width, crop_size
    
def findSun3DMetricCoords():
    
    cams = ['b3dodata', 'NYUdata', 'align_kv2', 'kinect2data', 'lg', 'sa', 'shr', 'sun3ddata', 'xtion_align_data']
    orig_metric_cord = {}
    crop_metric_cord = {}
    crop_sizes = {}
    
    for cam in cams:
        fx, fy ,cx, cy, height, width, crop_size = sun3Dintrinsic(cam)
        crop_sizes[cam] = crop_size
        final_fx, final_fy ,final_cx, final_cy = resizeIntrinsic(228, 304 ,fx, fy ,cx, cy, height/228) 
        orig_metric_cord[cam] = findMetricCordinates(final_fx, final_fy ,final_cx, final_cy, 304, 228)
    
        # 3 crops and resize all crops to 304x228
        crop1_fx, crop1_fy ,crop1_cx, crop1_cy = crop_and_resizeIntrinsic(crop_size[0][0], crop_size[0][1], height, width, fx, fy ,cx, cy, crop_size[0][0]/228 ) # 90% crop
        crop2_fx, crop2_fy ,crop2_cx, crop2_cy = crop_and_resizeIntrinsic(crop_size[1][0], crop_size[1][1], height, width,fx, fy ,cx, cy, crop_size[1][0]/228) # 80% crop
        crop3_fx, crop3_fy ,crop3_cx, crop3_cy = crop_and_resizeIntrinsic(crop_size[2][0], crop_size[2][1], height, width,fx, fy ,cx, cy, crop_size[2][0]/228) # 75% crop 
    
        crop_metric_cord[cam] = []
        crop_metric_cord[cam].append(findMetricCordinates(crop1_fx, crop1_fy ,crop1_cx, crop1_cy, 304, 228))
        crop_metric_cord[cam].append(findMetricCordinates(crop2_fx, crop2_fy ,crop2_cx, crop2_cy, 304, 228))
        crop_metric_cord[cam].append(findMetricCordinates(crop3_fx, crop3_fy ,crop3_cx, crop3_cy, 304, 228))
        
    return orig_metric_cord, crop_metric_cord, crop_sizes