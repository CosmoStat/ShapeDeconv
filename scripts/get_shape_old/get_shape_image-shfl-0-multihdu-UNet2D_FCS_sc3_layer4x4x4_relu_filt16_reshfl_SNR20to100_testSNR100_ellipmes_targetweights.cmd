#! /bin/bash
source /home/fsureau/.bashrc
/data/shapelens_v2/shapelens-CEA-master/bin/get_shapes -i /data/DeepDeconv/benchmark/euclidpsf/ellip/image-shfl-0-multihdu-target_fwhm0p07_weights.fits -p /data/DeepDeconv/data/gauss_fwhm0p07/starfield_image-000-0.fits -g 100 -s 96 -T /data/DeepDeconv/benchmark/euclidpsf/images/image-shfl-0-multihdu-UNet2D_FCS_sc3_layer4x4x4_relu_filt16_reshfl_SNR20to100_testSNR100.fits | tee /data/DeepDeconv/benchmark/euclidpsf/ellip/image-shfl-0-multihdu-UNet2D_FCS_sc3_layer4x4x4_relu_filt16_reshfl_SNR20to100_testSNR100_ellipmes_targetweights.txt