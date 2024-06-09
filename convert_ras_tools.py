import os
from qsub_utils import *

matlab_client = '/usr/local/MATLAB/R2013a_client'

flipnii2ras_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FlipNII2RAS.sh'
flipnii2ras_general_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FlipNII2RAS_General.sh'
fslfiletype_cmd = '/usr/local/fsl5/bin/fslchfiletype'
grad2fslgrad_cmd = '/ifs/loni/faculty/shi/spectrum/yxia/yihao_pipe/grad2fslgrad.sh'
dwi_split_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_SplitHCPDTIData.sh'



def copy_file_to_dir(tar_dir, filename):
    new_filename = os.path.join(tar_dir, os.path.basename(filename))
    os.system(f'cp {filename} {new_filename}')
    return new_filename


def convert_dwi_ras(input_dwi_file, input_bval_file, input_bvec_file, input_mask_file, data_pfix='data_ras', diff_dir=None, force_flg=False):
    
    '''
    Convert dwi data to ras space    
    '''
    if diff_dir is None:
        diff_dir = os.path.dirname(input_dwi_file)
    else:
        # copy dwi to local
        input_dwi_file = copy_file_to_dir(diff_dir, input_dwi_file)
        input_bval_file = copy_file_to_dir(diff_dir, input_bval_file)
        input_bvec_file = copy_file_to_dir(diff_dir, input_bvec_file)
        input_mask_file = copy_file_to_dir(diff_dir, input_mask_file)

    # standardlized output filenames
    output_gtable_file = f'{diff_dir}/{data_pfix}_gradtable.txt'
    output_dwi_ras_file = f'{diff_dir}/{data_pfix}.nii.gz'
    output_bval_file = f'{diff_dir}/{data_pfix}.bval'
    output_bvec_file = f'{diff_dir}/{data_pfix}.bvec'
    output_mask_ras_file = f'{diff_dir}/{data_pfix}_mask.nii.gz'

    print(output_dwi_ras_file)
    print(output_mask_ras_file)
    print(output_bval_file)
    print(output_bvec_file)

    if (not os.path.exists(output_dwi_ras_file)) or force_flg:
        
        # intermediate nii files
        input_dwi_nii_file = input_dwi_file.replace('.nii.gz', '.nii')
        input_mask_nii_file = input_mask_file.replace('.nii.gz', '.nii')

        output_dwi_ras_nii_file = output_dwi_ras_file.replace('.nii.gz', '.nii')
        output_mask_ras_nii_file = output_mask_ras_file.replace('.nii.gz', '.nii')
        
        # 1.1 covert dwi nii.gz to nii 
        os.system(f'{fslfiletype_cmd} NIFTI {input_dwi_file} {input_dwi_nii_file}')
        # 1.2 conver dwi nii to dwi_ras nii
        os.system(f'{flipnii2ras_cmd} {matlab_client} {input_dwi_nii_file} {output_dwi_ras_nii_file} {input_bval_file} {input_bvec_file} {output_gtable_file}')
        # 1.3 gzip dwi_ras nii
        os.system(f'{fslfiletype_cmd} NIFTI_GZ {output_dwi_ras_nii_file} {output_dwi_ras_file}')
        # 1.4 covert gtable to bval and bvec
        os.system(f'{grad2fslgrad_cmd} --gradt {output_gtable_file} --bval {output_bval_file} --bvec {output_bvec_file}')

        # 2.1 conver dwi mask to dwi mask ras nii
        os.system(f'{fslfiletype_cmd} NIFTI {input_mask_file} {input_mask_nii_file}')
        # 2.2 conver dwi mask nii to dwi_mask_ras nii
        os.system(f'{flipnii2ras_general_cmd} {matlab_client} {input_mask_nii_file} {output_mask_ras_nii_file}')
        # 2.3 gzip dwi_mask_ras nii
        os.system(f'{fslfiletype_cmd} NIFTI_GZ {output_mask_ras_nii_file} {output_mask_ras_file}')

        # clean up
        os.remove(input_dwi_nii_file)
        os.remove(output_dwi_ras_nii_file)
        os.remove(input_mask_nii_file)
        os.remove(output_mask_ras_nii_file)

    return output_dwi_ras_file, output_bval_file, output_bvec_file, output_mask_ras_file


def convert_vol_ras(input_file, output_file=None, data_pfix='_ras'):
    
    '''
    Convert volume data to ras space    
    '''
    if output_file is None:
        output_file = input_file.replace('.nii.gz', f'{data_pfix}.nii.gz')


    if not os.path.exists(output_file):

        # intermediate nii files
        # input_nii_file = input_file.replace('.nii.gz', '.nii')
        input_nii_file = output_file.replace('.nii.gz', '_input_temp.nii')
        output_nii_file = output_file.replace('.nii.gz', '.nii')

        # 2.1 conver dwi mask to dwi mask ras nii
        os.system(f'{fslfiletype_cmd} NIFTI {input_file} {input_nii_file}')
        # 2.2 conver dwi mask nii to dwi_mask_ras nii
        os.system(f'{flipnii2ras_general_cmd} {matlab_client} {input_nii_file} {output_nii_file}')
        # 2.3 gzip dwi_mask_ras nii
        os.system(f'{fslfiletype_cmd} NIFTI_GZ {output_nii_file} {output_file}')

        # clean up
        os.remove(input_nii_file)
        os.remove(output_nii_file)

    # if output_file is None:
    return output_file      


def mri_convert(input_file, output_file, qsub_flg=False):
    '''
    '''

    # convert_cmd = '/usr/local/freesurfer-5.3.0_64bit/bin/mri_convert'
    convert_cmd = '/ifs/loni/faculty/shi/spectrum/yxia/code/src/freesurfer_mri_convert.sh'

    convert_cmd_str = f'{convert_cmd} {input_file} {output_file}'
    job_id_num = randint(0,10000000)

    if qsub_flg:
        run7(f'mri_convert_{job_id_num}', convert_cmd_str)
    else:
        print(convert_cmd_str)
        os.system(convert_cmd_str)



# def skull_striping(input_file, output_file):
#     # extract 
#     input_file=/ifs/loni/faculty/shi/spectrum/yxia/dataset/HiRes_2023/processedDWI_session1_subset01/session1_subset01.nii.gz
#     b0_file = ${input_file//.nii.gz/_b0.nii.gz}
#     b0_n4corr_file = ${b0_file//.nii.gz/_n4corr.nii}
#     b0_n4corr_para_file = ${b0_file//.nii.gz/_n4corr_para.nii}

#     bet_file = ${b0_file//.nii.gz/_bet.nii.gz}
#     mask_file = ${input_file//.nii.gz/_mask.nii.gz}


#     /usr/local/fsl5/bin/fslroi $input_file $b0_file 0 1

#     /ifs/loni/faculty/shi/spectrum/Tools/N4BiasFieldCorrection -d 3 -i $b0_file -o '[' $b0_n4corr_file , $b0_n4corr_para_file ']'

#     /usr/local/fsl5/bin/bet $b0_n4corr_file $bet_file -f 0.1

#     /usr/local/fsl5/bin/fslmaths $bet_file -thr 50 -bin $mask_file