import os
import numpy as np
from qsub_utils import run7

matlab_client = '/usr/local/MATLAB/R2013a_client'
dwi_split_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_SplitHCPDTIData.sh'
fslfiletype_cmd = '/usr/local/fsl5/bin/fslchfiletype'
fslmerge_cmd = '/usr/local/fsl5/bin/fslmerge'

fslfiletype_cmd = '/ifs/loni/faculty/shi/spectrum/yxia/software/fsl/bin/fslchfiletype'
fslmerge_cmd = '/ifs/loni/faculty/shi/spectrum/yxia/software/fsl/bin/fslmerge'

def run_fslmerge(input_file_pattern, output_file, dim='z', force_flg=False, qsub_flg=False):
    if (not force_flg) | (not os.path.exists(output_file)):
        fslmerge_cmd_str = f'{fslmerge_cmd} -{dim} {output_file} {input_file_pattern}'

        if qsub_flg:
            run7('fslmerge', fslmerge_cmd_str, tmp_path=None)
        else:
            os.system(fslmerge_cmd_str)


# FOD
class FODParameters():
    def __init__(self, Max_order, Min_num_constraint, InitXi, xi_stepsize, LowBval=300, HighBval=3500, GradDev=0, Num_opti_steps=30, xi_max_numsteps=3, Max_num_fiber_crossings=4, 
                InitLambda1=0.0017, InitLambda2=0, Uniformity_flag=1, Noise_floor=0.0):
        self.Max_order = Max_order
        self.Min_num_constraint = Min_num_constraint
        self.InitXi = InitXi
        self.xi_stepsize = xi_stepsize
        self.LowBval = LowBval
        self.HighBval = HighBval
        self.GradDev = GradDev
        self.Num_opti_steps = Num_opti_steps
        self.xi_max_numsteps = xi_max_numsteps
        self.Max_num_fiber_crossings = Max_num_fiber_crossings
        self.InitLambda1 = InitLambda1
        self.InitLambda2 = InitLambda2
        self.Uniformity_flag = Uniformity_flag
        self.Noise_floor = Noise_floor

## HCP FOD Parameters
Max_order=16
Min_num_constraint=Max_order*10
InitXi=0.18 # (0.06 for 90)
xi_stepsize=np.round(InitXi/3, 4)
HCP_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize)

## HCP FOD Parameters
Max_order=12
Min_num_constraint=Max_order*10
InitXi=0.06 # (0.06 for 90) (47 1500b+ 46 3000b)
xi_stepsize=np.round(InitXi/3, 4)
HCPD_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize)

## HCP FOD Parameters
Max_order=12
Min_num_constraint=Max_order*10
InitXi=0.085 # (0.06 for 90) (64 1000b+ 64 2000b)
xi_stepsize=np.round(InitXi/3, 4)
HCP7T_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize)

## ABCD3000 FOD Parameters
Max_order=12
Min_num_constraint=Max_order*10
InitXi=0.04 # (0.06 for 90) (60 3000b)
xi_stepsize=np.round(InitXi/3, 4)
ABCD_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize)


Max_order=8
Min_num_constraint=Max_order*10
InitXi=0.03 # (0.06 for 90) (60 3000b)
xi_stepsize=np.round(InitXi/3, 4)
LowBval=800
HighBval=1200
adni_b1000_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize,
                                        LowBval=LowBval, HighBval=HighBval)

# single shell
Max_order=16
Min_num_constraint=Max_order*10
InitXi=0.09 # (0.06 for 90) (140 directions)
xi_stepsize=np.round(InitXi/3, 4)
LowBval=800
HighBval=3000
hr_sshell_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize,
                                        LowBval=LowBval, HighBval=HighBval)

# single shell section
Max_order=16
Min_num_constraint=Max_order*10
InitXi=0.18 # (0.06 for 90) (280 directions)
xi_stepsize=np.round(InitXi/3, 4)
LowBval=800
HighBval=3000
hr_sshellsection_FODParameters = FODParameters(Max_order, Min_num_constraint, InitXi, xi_stepsize,
                                        LowBval=LowBval, HighBval=HighBval)


FOD_PARA_DICT = {'HCPDevelopment':HCPD_FODParameters, 'HCP':HCP_FODParameters, 'HCP_7T':HCP7T_FODParameters, 
                'GE':ABCD_FODParameters, 'SIEMENS':ABCD_FODParameters}



def split_dwi(dwi_file, grad_dev_file, mask_file, output_dir, subj_id='subj_id', split_num=10):

    '''
    split dwi data input chunks
    '''

    dwi_nii_file = dwi_file.replace('.nii.gz', '.nii')
    mask_nii_file = mask_file.replace('.nii.gz', '.nii')
    grad_dev_nii_file = grad_dev_file.replace('.nii.gz', '.nii')

    os.system(f'{fslfiletype_cmd} NIFTI {dwi_file} {dwi_nii_file}')
    os.system(f'{fslfiletype_cmd} NIFTI {mask_file} {mask_nii_file}')
    if grad_dev_file != mask_file:
        os.system(f'{fslfiletype_cmd} NIFTI {grad_dev_file} {grad_dev_nii_file}')

    assert os.path.exists(dwi_nii_file)
    os.system(f'{dwi_split_cmd} {matlab_client} {dwi_nii_file} {grad_dev_nii_file} {mask_nii_file} {split_num} {output_dir} {subj_id}')

    # if split_num > 1:
    #     os.system(f'{dwi_split_cmd} {matlab_client} {dwi_nii_file} {grad_dev_nii_file} {mask_nii_file} {split_num} {output_dir} {subj_id}')
    # else:
    #     os.system(f'mkdir {output_dir}')
    #     os.system(f'cp {dwi_nii_file} {output_dir}')
    #     os.system(f'cp {mask_nii_file} {output_dir}')
    #     if os.path.exists(grad_dev_nii_file):
    #         os.system(f'cp {grad_dev_nii_file} {output_dir}')

    # clean up
    os.remove(dwi_nii_file)
    os.remove(mask_nii_file)
    if grad_dev_file != mask_file:
        os.remove(grad_dev_nii_file)


# FOD_kernel_org = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization.sh'
# FOD_MouseDTI_kernel = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization_MouseDTI.sh'
# FOD_MouseDTI_kernel2 = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization_MouseDTI2.sh'

def cal_fod_kernel(DWI_file, Mask_file, Grad_table, Output_FOD_file, Output_Tissuemapp_file,
                LowBval, HighBval, GradDev,
                Max_order, Min_num_constraint, Num_opti_steps, InitXi, xi_stepsize, xi_max_numsteps, Max_num_fiber_crossings, 
                InitLambda1=0.0017, InitLambda2=0, Uniformity_flag=1, Noise_floor=0.0,
                FOD_kernel='Org', matlab_ver = '/usr/local/MATLAB/MCR/R2013a_v81', job_location='cranium', job_name=None, job_log_dir=None):

                # minNumConstraint = spharm_order * 10
                # InitXi=0.2 for 270 directions 0.06 for 90 directions (HCPD have 90 directions)
                # xi_stepsize=xi_initialStep / 3

                if FOD_kernel == 'MouseDTI':
                    # the MouseDTI allow the {InitLambda1} and {InitLambda2}
                    FOD_kernel_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization_MouseDTI.sh'
                    ComStr = f'{FOD_kernel_cmd} {matlab_ver} {InitLambda1} {InitLambda2} {Grad_table} {LowBval} {HighBval} {DWI_file} {GradDev} {Mask_file}'
                    ComStr = f'{ComStr} {Max_order} {Min_num_constraint} {Num_opti_steps} {InitXi} {xi_stepsize} {xi_max_numsteps} {Max_num_fiber_crossings}'
                    ComStr = f'{ComStr} {Uniformity_flag} {Noise_floor} {Output_FOD_file} {Output_Tissuemapp_file}'
                elif FOD_kernel == 'MouseDTI2':
                    # the MouseDTI allow the {InitLambda1} and {InitLambda2}
                    FOD_kernel_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization_MouseDTI2.sh'
                    ComStr = f'{FOD_kernel_cmd} {matlab_ver} {InitLambda1} {InitLambda2} {Grad_table} {LowBval} {HighBval} {DWI_file} {GradDev} {Mask_file}'
                    ComStr = f'{ComStr} {Max_order} {Min_num_constraint} {Num_opti_steps} {InitXi} {xi_stepsize} {xi_max_numsteps} {Max_num_fiber_crossings}'
                    ComStr = f'{ComStr} {Uniformity_flag} {Noise_floor} {Output_FOD_file} {Output_Tissuemapp_file}'
                else:
                    FOD_kernel_cmd = '/ifs/loni/faculty/shi/spectrum/yshi/FOD/Code/run_FOD_AdaptiveConvexOpt_WholeVolume_KernelOptimization.sh'
                    ComStr = f'{FOD_kernel_cmd} {matlab_ver} {Grad_table} {LowBval} {HighBval} {DWI_file} {GradDev} {Mask_file}'
                    ComStr = f'{ComStr} {Max_order} {Min_num_constraint} {Num_opti_steps} {InitXi} {xi_stepsize} {xi_max_numsteps} {Max_num_fiber_crossings}'
                    ComStr = f'{ComStr} {Uniformity_flag} {Noise_floor} {Output_FOD_file} {Output_Tissuemapp_file}'

                print('*')
                print(ComStr)
                print(job_name)
                print(job_location)
                print(job_log_dir)

                if job_location == 'local':
                    os.system(ComStr)
                else:
                    if job_name is None:
                        job_name = 'FOD_' + os.path.splitext(os.path.basename(DWI_file))[0]

                    print(job_name)
                    # run7(job_name, ComStr, node='iniadmin7.q', tmp_path=job_log_dir)
                    run7(job_name, ComStr, h_vmem=16, node='compute7.q', tmp_path=job_log_dir)

