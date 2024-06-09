from utils import *
from diff_utils import *
from plumbum.cmd import ANTS,WarpImageMultiTransform,flirt
from plumbum import FG


def createAntsCaselist(fa_files, l0_files, file):

    with open(file,'w') as f:
        for FA, L0 in zip(fa_files, l0_files):
            f.write(f'{FA},{L0}\n')

# diffusion data registration

def warp_dwi(InputImg, RefImg, OutputImg, OutputMat, 
                InputDwiImg, OutputDwiImg, 
                InputBvec, OutputBvec, dof=6):

    "Also refer to DiffusionToStructural from HCPpipelines"

    # create output dir
    create_dir(dirname(OutputImg))
    create_dir(dirname(OutputMat))
    create_dir(dirname(OutputDwiImg))
    create_dir(dirname(OutputBvec))


    rotate_bvecs_cmd="/ifs/loni/faculty/shi/spectrum/yxia/Github/HCPpipelines/global/scripts/Rotate_bvecs.sh"

    # 1. estimate the transform mapping for dwi data
    flirt['-in', InputImg,
            '-ref', RefImg,
            '-out', OutputImg,
            '-omat', OutputMat,
            '-dof', dof] & FG


    # 2. apply the transform mapping for dwi data
    flirt['-in', InputDwiImg,
            '-ref', RefImg,
            '-applyxfm', 
            '-init', OutputMat,
            '-interp', 'spline',
            '-out', OutputDwiImg] & FG

    # 3. apply the transform mapping for bvec
    cmd_str = f'{rotate_bvecs_cmd} {InputBvec} {OutputMat} {OutputBvec}'
    os.system(cmd_str)


def forward_warp_flirt(InputImg, RefImg, AffineMat, OutputImg, InterpMode='spline'):
    ''' InterpMode: trilinear,nearestneighbour,sinc,spline'''

    flirt['-in', InputImg,
        '-ref', RefImg,
        '-applyxfm', 
        '-init', AffineMat,
        '-interp', InterpMode,
        '-out', OutputImg] & FG



def apply_SyN_MI(FIXED, MOVING, OUTPUTNAME, MAXITERATIONS,verbsome=False):
    
    if verbsome:
        print('SyN MI warpping ... ')
        print(f'FIXED: {FIXED}')
        print(f'MOVING: {MOVING}')

    
    METRIC='MI['    
    METRICPARAMS='1,32]'
    TRANSFORMATION='SyN[0.25]'
    REGULARIZATION='Gauss[3,0]'

    ANTS['3', 
        '-m', f'{METRIC}{FIXED},{MOVING},{METRICPARAMS}',
        '-t', TRANSFORMATION,
        '-r', REGULARIZATION,
        '-o', f'{OUTPUTNAME}_.nii.gz',
        '-i', MAXITERATIONS
        ] & FG
    
    warp_field = f'{OUTPUTNAME}_Warp.nii.gz'
    inverse_warp_field = f'{OUTPUTNAME}_InverseWarp.nii.gz'
    affine = f'{OUTPUTNAME}_Affine.txt'
    
    
    return warp_field,inverse_warp_field,affine


# def foreward_warp(moving_img, output_img, reference_img, warp_field, affine, verbsome=False):
#     if verbsome:
#         print(f'foreward warpping ... {moving_img} 2 {reference_img}')


#     WarpImageMultiTransform['3',  moving_img, output_img,
#                             '-R', reference_img,
#                             warp_field, affine] & FG    
    
    
# def backward_warp(reference_img, output_img, moving_img, inverse_warp_field, affine, verbsome=False):
#     if verbsome:
#         print('backward warpping ... ')
    

#     WarpImageMultiTransform['3',  reference_img, output_img,
#                             '-R', moving_img,
#                             '-i', affine, inverse_warp_field] & FG      


def foreward_warp(moving_img, output_img, reference_img, warp_field, affine, nn_flg=False, verbsome=False):
    if verbsome:
        print(f'foreward warpping ... {moving_img} 2 {reference_img}')

    if nn_flg:
        WarpImageMultiTransform['3',  moving_img, output_img,
                        '-R', reference_img, '--use-NN',
                        warp_field, affine] & FG    
    else:
        WarpImageMultiTransform['3',  moving_img, output_img,
                                '-R', reference_img,
                                warp_field, affine] & FG    
    
    
def backward_warp(reference_img, output_img, moving_img, inverse_warp_field, affine, nn_flg=False, verbsome=False):
    if verbsome:
        print('backward warpping ... ')
    
    if nn_flg:
        WarpImageMultiTransform['3',  reference_img, output_img,
                                '-R', moving_img, '--use-NN',
                                '-i', affine, inverse_warp_field] & FG        
    else:
        WarpImageMultiTransform['3',  reference_img, output_img,
                                '-R', moving_img,
                                '-i', affine, inverse_warp_field] & FG         


def backward_warpNN(reference_img, output_img, moving_img, inverse_warp_field, affine, verbsome=False):
    if verbsome:
        print('backward warpping ... ')
    

    WarpImageMultiTransform['3',  reference_img, output_img,
                            '-R', moving_img, '--use-NN',
                            '-i', affine, inverse_warp_field] & FG       



class mri_registration():
    '''
    registration objects for space transform

    '''
    def __init__(self, warp_field, inverse_warp_field, affine, template_img, sub_name=None, org_spacename=None, temp_spacename=None):
        '''
        Within the ANTs registration framework
        Input:
            warp_field: forward transformation filename
            inverse_warp_field: backward transformation filename
            warp_affine: forward transformation affine filename

            org_spacename:
            temp_spacename: 

        '''
        self.warp_field = warp_field
        self.inverse_warp_field = inverse_warp_field
        self.affine = affine
        self.template_img = template_img

        self.sub_name = sub_name
        self.org_spacename = org_spacename
        self.temp_spacename = temp_spacename
        

    def temp_forward(self, moving_img, output_img, reference_img, nn_flg=False, force_flg=False):
        '''
        Warp moving_img to reference_img space as output_img 
        Input:
            moving_img: moving image to warp
            output_img: output image in template space
            reference_img: template(reference) image in template space

        '''

        if not exists(output_img) or force_flg:
            if nn_flg:
                foreward_warp(moving_img, output_img, reference_img, self.warp_field, self.affine, nn_flg=nn_flg)
            else:
                foreward_warp(moving_img, output_img, reference_img, self.warp_field, self.affine, nn_flg=nn_flg)

 

    def temp_backward(self, reference_img, output_img, moving_img, nn_flg=False, force_flg=False):
        '''
        Backwarp reference_img to moving_img sace as output_img
        Input:
            reference_img: reference image (in template pace) for backwarping
            output_img: backwarped output (in subject pace)
            moving_img: image in subject space 

            nn_flg: nearist neighbor flag, True when warping Atlas back to subject space
        '''
        if not exists(output_img) or force_flg:
            if nn_flg:
                backward_warpNN(reference_img, output_img, moving_img, self.inverse_warp_field, self.affine)
            else:
                backward_warp(reference_img, output_img, moving_img, self.inverse_warp_field, self.affine)


def recent_nii(MRI_file, recent_MRI_file=None, force_flg=False):

    if recent_MRI_file is None:
        recent_MRI_file = add_fname_postfix(MRI_file, 'recent')

    if not exists(recent_MRI_file) or force_flg:

        MRI_nii = load(MRI_file)
        MRI_nii.get_fdata()
        MRI_nii.affine[:,3] = np.array([0,0,0,1]) # remove the offset
        
        save_nifti(recent_MRI_file, MRI_nii.get_fdata(), MRI_nii.affine, MRI_nii.header)

    return recent_MRI_file



def derecent_nii(MRI_file, recent_MRI_file, derecent_MRI_file, force_flg=False):

    if not exists(derecent_MRI_file) or force_flg:

        MRI_nii = load(MRI_file)
        MRI_nii.affine
        
        recent_MRI_nii = load(recent_MRI_file)
        recent_MRI_nii.get_fdata()
           
        save_nifti(derecent_MRI_file, recent_MRI_nii.get_fdata(), MRI_nii.affine, MRI_nii.header)