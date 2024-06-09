#!/bin/bash
export ANTSPATH=/usr/local/ANTs/bin/
export PATH=${ANTSPATH}:$PATH

outputPath=/ifs/loni/faculty/shi/spectrum/yxia/dataset_harm/HCP5/FA_template_recent_cran/
${ANTSPATH}antsMultivariateTemplateConstruction2.sh \
      -a 1 \
      -d 3 \
      -b 0 \
      -o ${outputPath}T_ \
      -g 0.2 \
      -c 1 \
      -i 6 \
      -k 1 \
      -f 8x4x2x1 \
      -n 0 \
      -r 1 \
      -t BSplineSyN[0.1,26,0] \
      -m CC \
      /ifs/loni/faculty/shi/spectrum/yxia/dataset_harm/HCP5/HC*/*/split_data/DTI_init/*_FA_recent.nii.gz

# /ifs/loni/faculty/shi/spectrum/yxia/dataset/HCP3/HC*/*/split_data/*_DTIVol_b3000_FA_recent.nii.gz

# -c:  Control for parallel computation (default 0):
#     0 = run serially
#     1 = SGE qsub
#     2 = use PEXEC (localhost)
#     3 = Apple XGrid
#     4 = PBS qsub
#     5 = SLURM