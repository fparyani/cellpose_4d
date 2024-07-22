#!/bin/bash 
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=2
#SBATCH --account pmg
#SBATCH --nodelist=m003

module load mamba
mamba init
source /burg/home/fp2409/.bashrc
mamba activate /pmglocal/fp2409/mambaforge/image_mask

python /pmglocal/fp2409/rotation_proj/analysis/preprocessing_cellpose/z_projection/run_cellpose_kernel_zcompression.py --czi_file /pmglocal/fp2409/embryo_img_kni_knrl/embryo03/ --output_dir /pmglocal/fp2409/rotation_proj/analysis/preprocessing_cellpose/z_projection/out/ --embryo_name embryo_3 --kernel 5 --z_compression max --cellpose_model_dimension 3D
