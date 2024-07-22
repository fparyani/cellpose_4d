# CellPose 4D Implementation

## Argument Description

`czi_file` file directory for where relevant czi files are placed. If there are multiple it will stitch them to together into one movie

`output_dir` is where you want the masks to be saved

`embryo_name` is a name you give to the file to be saved

`kernel` determines the size of the kernel to apply over x,y plane

`cellpose_model` determines which model to run, by default it is `nuclei` but optional arguments include `cyto` which runs the cytoplasm model. Optional argument

`z_compression` determines which compression to z_stack compression to apply. Options are `mean`, `max` , or `all` for both

`cellpose_model_dimension` is the CellPose model to run, can do either `2D` or `3D`

## Usage Example


```python
python /pmglocal/fp2409/rotation_proj/analysis/preprocessing_cellpose/z_projection/run_cellpose_kernel_zcompression.py --czi_file /pmglocal/fp2409/embryo_img_kni_knrl/embryo03/ --output_dir /pmglocal/fp2409/rotation_proj/analysis/preprocessing_cellpose/z_projection/out/ --embryo_name embryo_3 --kernel 5 --z_compression max --cellpose_model_dimension 3D
```

## Visualization/Analysis Script Usage

Load in masks output and czi image from above python script

```python

all_img_nuc = ReadData(czi_file_dir)

output_dir = "/Users/fahadparyani/Documents/Columbia/levo_lab/levo_lab/output/masks"
fname = os.path.join(output_dir, 'embryo_3_cellpose3D_nuclei_kernel5_masks.pkl')
with open(fname, 'rb') as f:
  masks3D_kern5 = pickle.load(f)
masks3D_kern5 = np.array(masks3D_kern5)
```


### Visualization of z_stack gif via `generate_4D_gif`

```python
generate_4D_gif(all_img_nuc[:,:,100:170,100:170], masks = masks3D_kern3[:,:,100:170,100:170],
                  pre_file_name = 'cellpose3D_kern3_subset', 
                  path = '/output/gif/')

#Shows both in different colors

generate_4D_gif_double(all_img_nuc[:,:,100:170,100:170], masks = masks3D_kern3[:,:,100:170,100:170], masks2 = masks3D_kern5[:,:,100:170,100:170],
                  pre_file_name = 'cellpose3D_kern3_v_kern5_subset', 
                  path = '/output/gif/')
```


### Volume measure 

`volume_measure(masks3D_kern5, file_dir = '/output/volume_measure/')`

### Feret Measurement

`mask_analysis(masks3D_kern5, z_ax_plane = 2, 
              file_dir = '/output/feret_diameter/', 
              file_name = 'masks3D_kern5_zax2_feret')
`
