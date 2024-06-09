'''
Util for General Medical Image Analysis of dMRI data

Yihao Xia (yihaoxia@usc.edu)
USC Viterbi School of Engineering 


Version 0.01
Last updated date: Sept. 20, 2023
'''

import warnings
import numpy as np
import pandas as pd
import skimage

from os.path import join as pjoin, split as psplit, abspath, basename, dirname, isfile, exists
from glob import glob
from shutil import copyfile, rmtree, copy2, copytree
from copy import deepcopy

from subprocess import check_call, Popen
import scipy.linalg as la

# Helpers from nibabel and dipy
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

import matplotlib.pyplot as plt


import importlib
def find_func(target_func_name, module_name):
    modellib = importlib.import_module(module_name)

    target_func = None
    for name, func in modellib.__dict__.items():
        if name.lower() == target_func_name.lower():
            target_func = func
            
    return target_func 

# Nifti Image I/O
def create_dir(_dir):
    if not exists(_dir):
        makedirs(_dir)      

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


from difflib import SequenceMatcher

def extr_common_postfix(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return string1[match.a: match.a + match.size]


def common_postfix(string_list):
    string_list = [basename(file) for file in string_list]
    
    common_string = string_list[0]
    for string in string_list[1:]:
        common_string = extr_common_postfix(common_string, string)
        
    return common_string


def add_fname_postfix(fname, postfix):
    '''Add postfix to the fname with a new subfolder'''
    nfname = fname.replace('.nii.gz',f'_{postfix}.nii.gz')
    nfname = pjoin(dirname(nfname), postfix, basename(nfname))
    return nfname

def add_fname_dirfix(fname, dirfix):
    '''Add postfix to the fname with a new subfolder'''
    nfname = pjoin(dirname(fname), dirfix, basename(fname))
    return nfname


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


def save_npz(npz_file, np_array, data_type='float16'):
    create_dir(dirname(npz_file))
    np.savez(npz_file, np_array)

def load_npz(npz_file, data_type='float16'):
    npzfile = np.load(npz_file)
    np_array = npzfile[npzfile.files[0]] 
    return np_array.astype(data_type) 


def linear_regression(y, X):
    # let voxel_number: v， sample_number: n， demo_feat_number: p
    # y: n*v，X: n*p， w: p*v， ref weight: n*n
    
    w = np.dot(np.dot(la.inv(np.dot(X.T, X)), X.T), y)

    return w


def statistic_outlier_threshold(score_array, scal=2):

    mean_score = np.mean(score_array)
    std_score = np.std(score_array)

    thr = mean_score + scal*std_score

    new_score_array = score_array[score_array <= thr]

    return new_score_array, thr


def iterative_statistic_outlier_threshold(score_array, scal=2, iter_num=10, diff_thr=0.001):

    score_array = score_array.flatten()

    thr_box = []
    new_score_array, thr = statistic_outlier_threshold(score_array, scal=scal)
    thr_box.append(thr)
    if iter_num > 0:
        for _ in range(iter_num):
            new_score_array, thr = statistic_outlier_threshold(new_score_array, scal=scal)
            thr_box.append(thr)
    else:
        diff = 999
        while diff > diff_thr:
            new_score_array, thr = statistic_outlier_threshold(new_score_array, scal=scal)
            thr_box.append(thr)
            diff = np.abs(thr_box[-1] - thr_box[-2])
    return thr_box


# to remove "parahippocampal" frontalpole

Frontal_label_list_old = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
            'parsopercularis', 'parstriangularis', 'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 
            'precentral', 'paracentral', 'frontalpole', 'rostralanteriorcingulate','caudalanteriorcingulate']
Temporal_label_list_old = ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'fusiform', 
                        'transversetemporal','entorhinal', 'temporalpole', 'parahippocampal']

Frontal_label_list = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
            'parsopercularis', 'parstriangularis', 'parsorbitalis', 
            'precentral', 'paracentral', 'frontalpole', 'rostralanteriorcingulate','caudalanteriorcingulate']

# Frontal_label_list = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
#             'parsopercularis', 'parstriangularis', 
#             'precentral', 'paracentral', 'frontalpole', 'rostralanteriorcingulate','caudalanteriorcingulate']
            
Temporal_label_list = ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'fusiform', 
                        'transversetemporal','entorhinal', 'parahippocampal']

# Frontal_label_list = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
#             'parsopercularis', 'parstriangularis', 'parsorbitalis', 
#             'precentral', 'paracentral', 'rostralanteriorcingulate','caudalanteriorcingulate']

# Temporal_label_list = ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'fusiform', 
#                         'transversetemporal','entorhinal']

Parietal_label_list = ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus']
Occipital_label_list = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine', 'posteriorcingulate',  'isthmuscingulate']
Cingulate_label_list = ['rostralanteriorcingulate','caudalanteriorcingulate', 'posteriorcingulate',  'isthmuscingulate']

lobe_mapping_dict_old = {'frontal': Frontal_label_list_old, 'parietal': Parietal_label_list, 'temporal': Temporal_label_list_old, 'occipital': Occipital_label_list}
lobe_mapping_dict = {'frontal': Frontal_label_list, 'parietal': Parietal_label_list, 'temporal': Temporal_label_list, 'occipital': Occipital_label_list}
lobe_mapping_dict = {'all': Frontal_label_list + Parietal_label_list + Temporal_label_list + Occipital_label_list + Cingulate_label_list, 
                    'frontal': Frontal_label_list, 'parietal': Parietal_label_list, 'temporal': Temporal_label_list, 'occipital': Occipital_label_list}



def dist2sim_exp(dist, dgamma=5, p=0.5):

    return np.exp(-dgamma*dist**p)
