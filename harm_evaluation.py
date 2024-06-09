from utils import *

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from nibabel import load, Nifti1Image
    from dipy.io import read_bvals_bvecs
    from dipy.io.image import load_nifti
    from dipy.core.gradients import gradient_table
    from dipy.reconst.shm import QballModel
    from dipy.reconst.odf import gfa
    from dipy.segment.mask import applymask
    import dipy.reconst.dti as reconst_dti
    
def save_nifti(fname, data, affine=None, hdr=None, ref_fname=None):

    if ref_fname is not None:
        ref_nii = load(ref_fname)
        affine = ref_nii.affine
        hdr = ref_nii.header

    else:
        if data.dtype.name=='uint8':
            hdr.set_data_dtype('uint8')
        else:
            hdr.set_data_dtype('float32')

    # create directory for nifti file if not exist
    create_dir(dirname(fname))

    result_img = Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


# preprocessing
def write_bvals(bval_file, bvals):
    with open(bval_file, 'w') as f:
        f.write(('\n').join(str(b) for b in bvals))


# Extract B0 and normalized dwi data
def find_b0(dwi, where_b0, mask=None):
    b0 = dwi[...,where_b0].mean(-1)
    np.nan_to_num(b0).clip(min= 0., out=b0) # inplace clip nans to zeros
    
    if mask is not None: # apply mask if available
        b0 = applymask(b0, mask > 0)
        
    return b0

def normalize_data(dwi, where_b0=None, mask=None, b0=None):

    dwi = dwi.astype('float32')
    if where_b0 is not None and b0 is None:
        b0 = find_b0(dwi, where_b0, mask)
        np.nan_to_num(b0).clip(min=1., out=b0) # inplace clip nans to ones
        for i in where_b0:
            dwi[...,i] = b0
    else:
        np.nan_to_num(b0).clip(min=1., out=b0) # inplace clip nans to ones
    
    # normalize dwi data by using b0
    # !!! numerical issue !!! dwi signal may be greater than b0 image
    dwiPrime = dwi/b0[...,None]
    np.nan_to_num(dwiPrime).clip(min=0., max=1., out=dwiPrime) # inplace clip normalized dwi data within [0,1]

    if mask is not None: # apply mask if available
        dwiPrime = applymask(dwiPrime, mask)

    return dwiPrime, b0

    
def img2mask(img_file, mask_file, thr=0):
    '''Binarize the image and save'''
    
    img_nii = load(img_file)
    img_nii.get_fdata() > thr
    
    save_nifti(mask_file, img_nii.get_fdata() > thr, img_nii.affine, img_nii.header)


def split_dwi_vol_z(dwi_file, slice_folder='slice_z', force_flg=False):
    '''
    split dwi along z slices
    '''

    # split dwi to slices and warp
    dwi_slice_files = []
    dwi_slice_file = add_fname_postfix(dwi_file, slice_folder)
    dwi_nii = load(dwi_file)
    dwi_img = dwi_nii.get_fdata()

    for i in range(dwi_img.shape[2]):
        adwi_slice_file = dwi_slice_file.replace('.nii.gz',f'{i}.nii.gz')

        if not exists(adwi_slice_file) or force_flg:
            dwi_slice = dwi_img[..., i, :]
            dwi_slice = np.expand_dims(dwi_slice, axis=2)
            save_nifti(adwi_slice_file, dwi_slice, affine=dwi_nii.affine, hdr=dwi_nii.header)

        dwi_slice_files.append(adwi_slice_file)

    return dwi_slice_files

def split_dwi_vol_d(dwi_file, slice_folder='slice_d', force_flg=False):
    '''
    split dwi along diffusion space
    '''

    # split dwi to slices and warp
    dwi_slice_files = []
    dwi_slice_file = add_fname_postfix(dwi_file, slice_folder)
    dwi_nii = load(dwi_file)
    dwi_img = dwi_nii.get_fdata()

    for i in range(dwi_img.shape[3]):
        adwi_slice_file = dwi_slice_file.replace('.nii.gz',f'{i}.nii.gz')

        if not exists(adwi_slice_file) or force_flg:
            dwi_slice = dwi_img[..., i]
#             dwi_slice = np.expand_dims(dwi_slice, axis=3)
            save_nifti(adwi_slice_file, dwi_slice, affine=dwi_nii.affine, hdr=dwi_nii.header)

        dwi_slice_files.append(adwi_slice_file)

    return dwi_slice_files


def merge_slice(slice_files, merge_file, axis=3):
    '''
    Merge slices along axis=n

    '''
    merged_img = np.moveaxis(np.array([load(slice_file).get_fdata() for slice_file in slice_files]), 0, axis)
    
    save_nifti(merge_file, merged_img, ref_fname=slice_files[0])


