import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import czifile
import pickle
import os
import re
import imageio
from skimage import io, color, measure, morphology
from scipy.ndimage import convolve
from tqdm import tqdm
from cellpose import models, io, utils, plot
from PIL import Image, ImageDraw, ImageFont
from skimage import img_as_ubyte
import seaborn as sns
from skimage.measure import label, regionprops,regionprops_table

def decompress_z_stack(compressed_image, original_z_dim, group_size=3):
    time_dim, compressed_z_dim, x_dim, y_dim = compressed_image.shape
    
    # Initialize an array for the decompressed image
    decompressed_image = np.zeros((time_dim, original_z_dim, x_dim, y_dim))
    
    # Fill in the decompressed image
    for i in range(compressed_z_dim):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        
        if end_idx <= original_z_dim:
            decompressed_image[:, start_idx:end_idx, :, :] = compressed_image[:, i:i+1, :, :]
        else:
            # Handle the case where the last slices don't fit perfectly into groups of 3
            decompressed_image[:, start_idx:original_z_dim, :, :] = compressed_image[:, i:i+1, :, :]
    
    return decompressed_image

def combine_gifs(gif_paths, grid_shape, output_path, default_duration=100, border_thickness=5, border_color=(255, 255, 255)):
    gifs = [imageio.mimread(gif) for gif in gif_paths]
    
    try:
        gif_duration = imageio.get_reader(gif_paths[0]).get_meta_data().get('duration', default_duration)
    except KeyError:
        gif_duration = default_duration

    num_frames = len(gifs[0])
    gif_height, gif_width, _ = gifs[0][0].shape

    grid_height, grid_width = grid_shape
    combined_gif = []

    for frame_index in range(num_frames):
        new_frame = Image.new('RGB', ((grid_width * (gif_width + border_thickness)), (grid_height * (gif_height + border_thickness))), border_color)
        for i, gif in enumerate(gifs):
            row = i // grid_width
            col = i % grid_width
            frame = Image.fromarray(gif[frame_index])
            
            # Add border to the frame
            bordered_frame = Image.new('RGB', (gif_width + border_thickness, gif_height + border_thickness), border_color)
            bordered_frame.paste(frame, (border_thickness // 2, border_thickness // 2))
            
            # Draw time point text on the frame
            draw = ImageDraw.Draw(bordered_frame)
            font = ImageFont.load_default()
            text = f"T = {frame_index}"
            draw.text((10, 10), text, font=font, fill=(255, 0, 0))  # Red color
            
            new_frame.paste(bordered_frame, (col * (gif_width + border_thickness), row * (gif_height + border_thickness)))
        combined_gif.append(new_frame)

    combined_gif[0].save(output_path, save_all=True, append_images=combined_gif[1:], duration=gif_duration, loop=0)


def double_overlay(mask1,mask2, image_in):
    # Convert the image to RGB if not already
    img0 = plot.image_to_rgb(image_in.copy(), channels=[0, 0])

    # Create a copy of the image to overlay the masks
    overlay_image = img0.copy()

    # Define colors for the masks
    color_mask1 = np.array([36, 255, 12])    # Green for mask1
    color_mask2 = np.array([255, 128, 0])    # Orange for mask2

    # Overlay Mask1: Add color to the mask areas
    overlay_image[mask1 == 1] = color_mask1

    # Overlay Mask2: Add color to the mask areas, handling overlap
    overlay_image[mask2 == 2] = color_mask2

    # Ensure the image is in uint8 format for display
    overlay_image = img_as_ubyte(overlay_image)

    return overlay_image

def mask_overlay(image_in, mask_in, line_thick = 1, color_label = 1):
    
    mask = np.ones(viz_masks(mask_in).shape, dtype=np.uint8) * 255
    
    gray = cv2.cvtColor(plot.image_to_rgb(viz_masks(mask_in), channels=[0,0]), cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (36, 255, 12), thickness=line_thick)
        
    mask[np.where(mask == 255)] = 0
    mask[np.where(mask != 0)] = color_label

    color_scheme = [(36, 255, 12), #green = 1 label
                (255,128,0), # orange/yellow = 2 mask
                (0,255,0), # green = 3
                (0,128,255), # blue = 4
                (0,0,255), #darker blue = 5
                (255,0,255)] # purple = 6

    img0 = plot.image_to_rgb(image_in.copy(), channels=[0,0])
    
    overlay_image = color.label2rgb(mask, img0, colors=color_scheme, alpha=1, bg_label=0, bg_color=None)

    overlay_image = (overlay_image * 255).astype(np.uint8)

    return overlay_image, mask


def generate_4D_gif_double(img4d, masks, masks2, pre_file_name, path, 
                    z_ax_range = np.arange(0,20,2,), grid_shape_in = (2,5), line_thick_in = 1):
    
    if not os.path.exists(path):
        os.makedirs(path)

    for z_dim in z_ax_range:
        overlayed_images = []
         # Process each time point
        for t in range(img4d.shape[0]):

            hold_image, mask_out1 = mask_overlay(img4d[t, z_dim, :, :], masks[t][z_dim], line_thick=line_thick_in)
            hold_image, mask_out2 = mask_overlay(img4d[t, z_dim, :, :], masks2[t][z_dim], line_thick=line_thick_in, color_label = 2)

            overlayed_image = double_overlay(mask_out1, mask_out2, img4d[t, z_dim, :, :])
            overlayed_images.append(overlayed_image)

            
            # overlayed_image = mask_overlay(img4d[t,z_dim,:,:], masks[t][z_dim], line_thick = line_thick_in)
            # overlayed_images.append(overlayed_image)
    
        overlayed_images = np.array(overlayed_images)
        
        imageio.mimsave(path+pre_file_name+'_z'+str(z_dim)+'.gif', overlayed_images, duration = 5)

    # Combine all z-slice GIFs into a grid
    gif_paths = [f"{path}{pre_file_name}_z{z}.gif" for z in z_ax_range]

    combine_gifs(gif_paths, grid_shape=grid_shape_in, output_path= path+pre_file_name+'_combined_z_slices.gif')

    return print("done")

def generate_4D_gif(img4d, masks, masks2, pre_file_name, path, 
                    z_ax_range = np.arange(0,20,2,), grid_shape_in = (2,5), line_thick_in = 1):
    
    if not os.path.exists(path):
        os.makedirs(path)

    for z_dim in z_ax_range:
        overlayed_images = []
         # Process each time point
        for t in range(img4d.shape[0]):


            
            overlayed_image = mask_overlay(img4d[t,z_dim,:,:], masks[t][z_dim], line_thick = line_thick_in)
            overlayed_images.append(overlayed_image)
    
        overlayed_images = np.array(overlayed_images)
        
        imageio.mimsave(path+pre_file_name+'_z'+str(z_dim)+'.gif', overlayed_images, duration = 5)

    # Combine all z-slice GIFs into a grid
    gif_paths = [f"{path}{pre_file_name}_z{z}.gif" for z in z_ax_range]

    combine_gifs(gif_paths, grid_shape=grid_shape_in, output_path= path+pre_file_name+'_combined_z_slices.gif')

    return print("done")

def volume_measure(masks_mat, file_dir = None, file_name = None):
    store_volume = []
    time_pt = masks_mat.shape[0]

    # for time in np.arange(0,time_pt,1):
    for time in tqdm(np.arange(0,time_pt,1), desc="Processing Time Points"):

        num_masks = len(np.unique(masks_mat[time,]))
        hold_mask_vol = []
        for mask in np.arange(1,num_masks,1):
            
            
            mask_volume = len(np.where(masks_mat[time,:,:,:] == mask)[0])
            hold_mask_vol.append({'time': time, 'mask': mask, 'volume': mask_volume})

        store_volume.extend(hold_mask_vol)

    df = pd.DataFrame(store_volume)

    if file_dir != None:
        
        # fname = os.path.join(file_dir,file_name +'.pkl')
        fname = os.path.join(file_dir,file_name +'.csv')

        df.to_csv(fname)
        
        # with open(fname, 'wb') as f:
        #     pickle.dump(store_volume,f)
        
        

    return store_volume

def feret_measure(masks_mat, z_ax_plane=None, file_dir=None, file_name=None):
    """
    Computes the Feret Diameter across time for one Z_ax and stores the data in a Pandas DataFrame.
    """
    if z_ax_plane is not None:
        store_feret_diam = []
        time_pt = masks_mat.shape[0]

        for time in tqdm(np.arange(0, time_pt, 1), desc="Processing Time Points"):
            properties = regionprops_table(masks_mat[time, z_ax_plane, :, :], properties=('label', 'feret_diameter_max'))
            for label, feret_diam in zip(properties['label'], properties['feret_diameter_max']):
                store_feret_diam.append({'time': time, 'label': label, 'feret_diameter_max': feret_diam})

        df = pd.DataFrame(store_feret_diam)

        if file_dir is not None:
            fname = os.path.join(file_dir, file_name + '.csv')
            df.to_csv(fname)

        return df
    else:
        return pd.DataFrame()


def gen_single_gif(image_max, masks_max, output_path, default_duration=100, line_thick_in = 1):

    time_pt = image_max.shape[0]
        
    
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    
    image_series = []
    for t in range(time_pt):
        overlay_image, mask_hold = mask_overlay(image_max[t,:,:], masks_max[t], line_thick = line_thick_in)
        image_series.append(overlay_image)
        
    num_frames = len(image_series)
    gif_height, gif_width = image_series[0].shape[:2]

    overlayed_images = []

    for frame_index in range(num_frames):
        frame = Image.fromarray(image_series[frame_index])

        # Draw time point text on the frame
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default()
        text = f"T = {frame_index}"
        draw.text((10, 10), text, font=font, fill=(255, 0, 0))  # Red color

        overlayed_images.append(np.array(frame))

    # Save the GIF using imageio.mimsave
    imageio.mimsave(output_path, overlayed_images, duration=default_duration)
