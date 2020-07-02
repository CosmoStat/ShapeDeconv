#! /bin/bash
source /home/fsureau/.bashrc
/data/shapelens_v2/shapelens-CEA-master/bin/get_shapes -i /data/DeepDeconv/benchmark/euclidpsf/ellip/image-shfl-0-multihdu-target_fwhm0p07_weights.fits -p /data/DeepDeconv/data/gauss_fwhm0p07/starfield_image-000-0.fits -g 100 -s 96 -T /data/DeepDeconv/benchmark/euclidpsf/images/image-shfl-0-multihdu-DeconvNetRef_FCS_train2to20_val1_vsc_rangeSNR20to100_testSNR40.fits | tee /data/DeepDeconv/benchmark/euclidpsf/ellip/image-shfl-0-multihdu-DeconvNetRef_FCS_train2to20_val1_vsc_rangeSNR20to100_testSNR40_ellipmes_targetweights.txt