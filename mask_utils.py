# from utils import *
from glob import glob
from diff_utils import *

from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes


def refine_brainmask(mask_img):

    mask_img = binary_dilation(mask_img)
    mask_img = binary_dilation(mask_img)
    mask_img = binary_erosion(mask_img)
    mask_img = binary_erosion(mask_img)
    mask_img = binary_dilation(mask_img)
    mask_img = binary_erosion(mask_img)
    mask_img = binary_erosion(mask_img)
    mask_img = binary_dilation(mask_img)
    mask_img = binary_fill_holes(mask_img).astype(int)

    return mask_img

def template_space_frame(template_dir, template_thr=0.001, refine_flg=False, verbsome=True):
    '''
    Generate the image frame in the template space
    1. Return the affine and header of template
    2. Return the dwi mask of the template
    '''
    
    template_file = f'{template_dir}/T_template0.nii.gz'
    tmp_affine = load(template_file).affine
    tmp_header = load(template_file).header

    if refine_flg:
        template_mask_file = f'{template_dir}/template_img_mask_{template_thr}_refined.nii.gz'
    else:
        template_mask_file = f'{template_dir}/template_img_mask_{template_thr}.nii.gz'

    if exists(template_mask_file):
        if verbsome:
            print(f'Using existing mask file {template_mask_file}')
        template_mask_img = load(template_mask_file).get_fdata() > 0
    else:
        sub_files = sorted(glob(f'{template_dir}/T_template0*WarpedToTemplate.nii.gz'))
        if verbsome:
            print(f'Creating mask file {template_mask_file} from {len(sub_files)} subjects ')

        sub_imgs = np.array([load(sub_file).get_fdata() for sub_file in sub_files])
        template_mask_img = np.min(sub_imgs,axis=0) > template_thr

        if refine_flg:
            template_mask_img = refine_brainmask(template_mask_img)

        save_nifti(template_mask_file, template_mask_img, tmp_affine, hdr=tmp_header)


    return template_file, template_mask_img, template_mask_file, tmp_affine, tmp_header