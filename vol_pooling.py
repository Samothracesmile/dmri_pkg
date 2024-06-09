from utils import *
from diff_utils import *


def mean_nii(img_files, output_filename=None, mask_img=None):
    '''Calculate the mean image of list of nifty files'''

    data_stack = np.array([load(img).get_fdata() for img in img_files])
    mean_data = np.mean(data_stack,0)

    if mask_img is not None:
        mean_data[~mask_img] = 0
    
    if output_filename is not None:
        hdr = load(img_files[0]).header
        affine = load(img_files[0]).affine
        save_nifti(output_filename, mean_data, affine, hdr)
    
    return mean_data


def median_nii(img_files, output_filename=None, mask_img=None):
    '''Calculate the med image of list of nifty files'''

    data_stack = np.array([load(img).get_fdata() for img in img_files])
    med_data = np.median(data_stack,0)

    if mask_img is not None:
        mean_data[~mask_img] = 0
    
    if output_filename is not None:
        hdr = load(img_files[0]).header
        affine = load(img_files[0]).affine
        save_nifti(output_filename, med_data, affine, hdr)
    
    return med_data


def std_nii(img_files, output_filename=None, mask_img=None):

    data_stack = np.array([load(img).get_fdata() for img in img_files])
    std_data = np.std(data_stack,0)

    if mask_img is not None:
        cov_data[~mask_img] = 0

    if output_filename:
        hdr = load(img_files[-1]).header
        affine = load(img_files[-1]).affine
        save_nifti(output_filename, std_data, affine, hdr)

    return std_data


def cov_nii(img_files, output_filename=None, mask_img=None):
    '''Coefficient of variance'''
    
    data_stack = np.array([load(img).get_fdata() for img in img_files])
    cov_data = np.std(data_stack,0)
    mean_data = np.mean(data_stack,0)

    cov_data[mean_data > 0.0000001] = cov_data[mean_data > 0.0000001]/mean_data[mean_data > 0.0000001]
    cov_data[mean_data <= 0.0000001] = 0

    if mask_img is not None:
        cov_data[~mask_img] = 0

    if output_filename:
        hdr = load(img_files[-1]).header
        affine = load(img_files[-1]).affine
        save_nifti(output_filename, cov_data, affine, hdr)
    
    return cov_data