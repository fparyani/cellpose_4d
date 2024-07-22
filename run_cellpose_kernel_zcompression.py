import matplotlib.pyplot as plt
import numpy as np
from cellpose import models, io
from cellpose.io import imread
from cellpose import utils
from cellpose import plot
import czifile
import pickle
import os
import pandas as pd
import re
# from skimage import io, colora
from tqdm import tqdm
from scipy.ndimage import convolve
from skimage import io

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def ReadData(dir):
    files = os.listdir(dir)

    sorted_files = sorted(files, key=extract_number)

    image_arrays = [czifile.imread(dir+file)[0,0,:,:,:,:,:,0] for file in sorted_files]
    merged_img = np.concatenate(image_arrays, axis = 1)

    return merged_img

def compress_z_stack(image, group_size=3):
    time_dim, z_dim, x_dim, y_dim = image.shape
    
    # Determine the number of complete groups of `group_size`
    num_full_groups = z_dim // group_size
    remaining_slices = z_dim % group_size

    # Reshape to separate into full groups of `group_size`
    full_groups_shape = (time_dim, num_full_groups, group_size, x_dim, y_dim)
    full_groups = image[:, :num_full_groups*group_size, :, :].reshape(full_groups_shape)

    # Calculate the average and max projections for full groups
    average_projection = full_groups.mean(axis=2)
    max_projection = full_groups.max(axis=2)
    
    # Handle remaining slices if any
    if remaining_slices > 0:
        remaining_slices_data = image[:, -remaining_slices:, :, :]
        remaining_average = remaining_slices_data.mean(axis=1, keepdims=True)
        remaining_max = remaining_slices_data.max(axis=1, keepdims=True)

        # Append the remaining slices' projections to the results
        average_projection = np.concatenate((average_projection, remaining_average), axis=1)
        max_projection = np.concatenate((max_projection, remaining_max), axis=1)

    return average_projection, max_projection


def ParseArgs():
    import argparse

    parser = argparse.ArgumentParser(description='4D Image Segmention')

    parser.add_argument('--czi_file', required=True,
                    help='the path to the raw CZI file')

    parser.add_argument('--output_dir', required=True,
                    help='path to place output')

    parser.add_argument('--embryo_name', required=True,
                    help='path to place output')

    parser.add_argument('--kernel', type = int, required=True,  
                    help='size of kernel for convolution size:(n X n)')

    parser.add_argument('--cellpose_model', required=False, default='nuclei',  
                    help='which cellpose model type to run')

    parser.add_argument('--z_projection', required=False, default = 'all',  
                    help='options to run are "mean" or "max" will run both by default')

    parser.add_argument('--cellpose_model_dimension', required=True,  
                    help='options to run are "3D" or "2D"')

    

    args = parser.parse_args()
    print('Arguments list:')
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print()

    return args


def main(args):
    all_img = ReadData(args.czi_file)
    all_img_nuc = all_img[1,:,:,:,:]
    
    all_img_nuc_smooth = all_img_nuc.copy()

    kernel = np.ones((args.kernel, args.kernel)) / (args.kernel * args.kernel)

     # Iterate over the first two dimensions
    for i in range(all_img_nuc.shape[0]):
        for j in range(all_img_nuc.shape[1]):
            # Apply the convolution to each 2D slice
            all_img_nuc_smooth[i, j, :, :] = convolve(all_img_nuc[i, j, :, :], kernel)

    average_proj, max_proj = compress_z_stack(all_img_nuc_smooth)
        

    #Inputs CellPose Model
    model = models.Cellpose(model_type=args.cellpose_model)
    channels = [0,0]
    num_nuc_store = []
    
    if args.cellpose_model_dimension == "3D":

        if args.z_projection == "all":
            store_mask_mean = []
            store_mask_max = []
            for i in np.arange(0,all_img_nuc_smooth.shape[0],1):

                #Runs CellPose
                masks_mean, flows_mean, styles_mean, diams_mean = model.eval(average_proj[i,:,:,:], channels=channels, do_3D = True)
                store_mask_mean.append(masks_mean)

            for i in np.arange(0,all_img_nuc_smooth.shape[0],1):

                #Runs CellPose
                masks_max, flows_max, styles_max, diams_max = model.eval(max_proj[i,:,:,:], channels=channels, do_3D = True)
                store_mask_max.append(masks_max)


            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose3D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_mean'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_mean,f)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose3D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_max'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_max,f)            

        elif args.z_projection == "mean":
            store_mask_mean = []

            for i in np.arange(0,all_img_nuc_smooth.shape[0],1):

                #Runs CellPose
                masks_mean, flows_mean, styles_mean, diams_mean = model.eval(average_proj[i,:,:,:], channels=channels, do_3D = True)
                store_mask_mean.append(masks_mean)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose3D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_mean'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_mean,f)

        elif args.z_projection == "max":
            store_mask_max = []
            for i in np.arange(0,all_img_nuc_smooth.shape[0],1):

                #Runs CellPose
                masks_max, flows_max, styles_max, diams_max = model.eval(max_proj[i,:,:,:], channels=channels, do_3D = True)
                store_mask_max.append(masks_max)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose3D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_max'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_mean,f)

        # store_mask = []
        # for i in np.arange(0,all_img_nuc_smooth.shape[0],1):

        #     #Runs CellPose
        #     masks3D, flows3D, styles3D, diams3D = model.eval(all_img_nuc_smooth[i,:,:,:], channels=channels, do_3D = True)
        #     store_mask_zax.append(masks3D)


    if args.cellpose_model_dimension == "2D":

        if args.z_projection == "all":
            store_mask_mean = []
            store_mask_max = []

            for i in np.arange(0,average_proj.shape[0],1):
                seplist = []
                for j in np.arange(0,average_proj.shape[1],1):

                    masks_mean, flows_mean, styles_mean, diams_mean = model.eval(average_proj[i, j, :, :], diameter=None, channels=channels)
                    seplist.append(masks_mean)

                store_mask_mean.append(seplist)

            for i in np.arange(0,max_proj.shape[0],1):
                seplist = []
                for j in np.arange(0,max_proj.shape[1],1):

                    masks_max, flows_max, styles_max, diams_max = model.eval(max_proj[i, j, :, :], diameter=None, channels=channels)
                    seplist.append(masks_max)

                store_mask_max.append(seplist)


            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose2D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_mean'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_mean,f)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose2D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_max'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_max,f)    


        elif args.z_projection == "mean":

            store_mask_mean = []

            for i in np.arange(0,average_proj.shape[0],1):
                seplist = []
                for j in np.arange(0,average_proj.shape[1],1):

                    masks_mean, flows_mean, styles_mean, diams_mean = model.eval(average_proj[i, j, :, :], diameter=None, channels=channels)
                    seplist.append(masks_mean)

                store_mask_mean.append(seplist)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose2D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_mean'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_mean,f)

        elif args.z_projection == "max":
            store_mask_max = []

            for i in np.arange(0,max_proj.shape[0],1):
                seplist = []
                for j in np.arange(0,max_proj.shape[1],1):

                    masks_max, flows_max, styles_max, diams_max = model.eval(max_proj[i, j, :, :], diameter=None, channels=channels)
                    seplist.append(masks_max)

                store_mask_max.append(seplist)

            fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose2D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_max'+'_masks.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(store_mask_max,f)   
   
    # fname = os.path.join(args.output_dir, args.embryo_name+'_cellpose3D_'+args.cellpose_model+'_kernel'+str(args.kernel)+'_masks.pkl')
    # with open(fname, 'wb') as f:
    #     pickle.dump(store_mask_zax,f)

    





if __name__ == "__main__":
  args = ParseArgs()
  main(args)


    
