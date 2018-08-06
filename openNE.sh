#!/bin/bash
#$ -l rmem=4G
#$ -P tale
#$ -q tale.q

module load apps/python/anaconda3-4.2.0
source activate myexperiment

# Download 1 4D NIfTI image for illustration
#python download_abide_preproc.py -d func_preproc -p cpac -s filt_global -o "func_img"
#python download_abide_preproc.py -d func_preproc -p niak -s filt_global -o "func_img"

# Download the ROI time-series data
python src/main.py --method line --label-file data/wiki/Wiki_category.txt --input data/wiki/Wiki_edgelist.txt --graph-format edgelist --output vec_all.txt --order 1 --epoch 20 --clf-ratio 0.5ΩΩ

# TODO: Download the atlases automatically?
