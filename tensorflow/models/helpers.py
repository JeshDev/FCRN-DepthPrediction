import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
import h5py

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
    
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    crop_cx_rgb = cx_rgb - 8 #because of white border cropping
    crop_cy_rgb = cy_rgb - 6  
    
    cropped_width = 624
    cropped_height = 468
    scale = 0.48717
    new_fx, new_fy ,new_cx, new_cy = resizeIntrinsic(cropped_height, cropped_width, fx_rgb, fy_rgb, crop_cx_rgb, crop_cy_rgb, scale) #resized to (228, 304, 3)
    
    return new_fx, new_fy ,new_cx, new_cy
    
def crop_and_resizeIntrinsic(crop_height, crop_width, orig_height, orig_width, fx_rgb, fy_rgb, cx_rgb, cy_rgb, scale):
    
    #new_cx_rgb = cx_rgb + float(width-1)/2 - crop_col_cj
    #new_cy_rgb = cy_rgb + float(height-1)/2 - crop_row_ci
    offset_cx = (orig_height - crop_height)/2
    offset_cy = (orig_width - crop_width)/2
    new_cx_rgb = cx_rgb - offset_cx
    new_cy_rgb = cy_rgb - offset_cy
    new_fx, new_fy ,new_cx, new_cy = resizeIntrinsic(crop_height, crop_width, fx_rgb, fy_rgb, new_cx_rgb, new_cy_rgb, scale)
    return new_fx, new_fy ,new_cx, new_cy

def resizeIntrinsic(height, width, fx, fy, cx_rgb, cy_rgb, scale):
    
    center_x = float(width-1) / 2
    center_y = float(height-1) / 2
    orig_cx_diff = cx_rgb - center_x
    orig_cy_diff = cy_rgb - center_y
    scaled_height = round(scale * height)
    scaled_width = round(scale * width)
    scaled_center_x = float(scaled_width-1) / 2
    scaled_center_y = float(scaled_height-1) / 2
    new_fx = scale * fx
    new_fy = scale * fy
    #skew = scale * self.skew
    new_cx = scaled_center_x + scale * orig_cx_diff
    new_cy = scaled_center_y + scale * orig_cy_diff
    return new_fx, new_fy ,new_cx, new_cy
    
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], x, y
