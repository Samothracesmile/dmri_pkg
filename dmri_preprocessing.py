'''
Calculate dMRI preprocessing:  1.denoise, 2.unring

'''
from utils import *
from diff_utils import *
from dipy.denoise.localpca import localpca, mppca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.denoise.patch2self import patch2self
from dipy.denoise.gibbs import gibbs_removal
import os

FSLDIR = os.environ['FSLDIR']

def fsl_flirt(input_file, ref_file, output_file, tar_res, mask_flg=False):
    
    flirt_cmd = f'{FSLDIR}/bin/flirt -in {input_file} -ref {ref_file} -out {output_file} -datatype float -applyisoxfm {tar_res}'
    os.system(flirt_cmd)
    if mask_flg:
        flirt_cmd = f'{FSLDIR}/bin/fslmaths {output_file} -thr 0.75 -bin {output_file}'   
        os.system(flirt_cmd)


def denoise_dwi_all(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Smoothing the dwi using the dipy local pca + gibbs_removal
    '''

    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_mask = load(dwi_mask_file).get_fdata() > 0
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)

        dwi = gibbs_removal(dwi, slice_axis=2) # gibbs correction along z-axis
        sigma = pca_noise_estimate(dwi, gtab, correct_bias=True, smooth=3)
        denoised_dwi = localpca(dwi, sigma, mask=dwi_mask, tau_factor=2.3, patch_radius=2)
        denoised_dwi = applymask(denoised_dwi, mask)

        save_nifti(denoised_dwi_file, denoised_dwi, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - denoised_dwi

            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)



def denoise_dwi_gibbs(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Gibbs unring by using dipy
    '''

    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_mask = load(dwi_mask_file).get_fdata() > 0
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)
        
        dwi_corrected = gibbs_removal(dwi, slice_axis=2) # gibbs correction along z-axis
        dwi_corrected = applymask(dwi_corrected, mask)
        save_nifti(denoised_dwi_file, dwi_corrected, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - dwi_corrected
            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)


def denoise_dwi_localpca(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Smoothing the dwi using the dipy local pca
    '''

    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_mask = load(dwi_mask_file).get_fdata() > 0
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)
        
        sigma = pca_noise_estimate(dwi, gtab, correct_bias=True, smooth=3)
        denoised_dwi = localpca(dwi, sigma, mask=dwi_mask, tau_factor=2.3, patch_radius=2)
        denoised_dwi = applymask(denoised_dwi, mask)
        save_nifti(denoised_dwi_file, denoised_dwi, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - denoised_dwi
            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)


def denoise_dwi_mppca(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Smoothing the dwi using the dipy local mppca
    '''

    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_mask = load(dwi_mask_file).get_fdata() > 0
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)


        denoised_dwi = mppca(dwi, mask=dwi_mask, patch_radius=2, pca_method='eig',
                              return_sigma=False, out_dtype=None)
    
        denoised_dwi = applymask(denoised_dwi, mask)
        save_nifti(denoised_dwi_file, denoised_dwi, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - denoised_dwi
            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)




def denoise_dwi_native_mppca(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Smoothing the dwi using the dipy using native mppca
    '''

    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_mask = load(dwi_mask_file).get_fdata() > 0
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)

        from mppca_denoise.mpdenoise import denoise
        denoised_dwi, _, _ = denoise(dwi, kernel='5,5,5')
        # denoised_dwi = mppca(dwi, mask=dwi_mask, patch_radius=2, pca_method='eig',
        #                       return_sigma=False, out_dtype=None)
    
        denoised_dwi = applymask(denoised_dwi, mask)
        save_nifti(denoised_dwi_file, denoised_dwi, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - denoised_dwi
            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)


def denoise_dwi_patch2self(dwi_file, bval_file, bvec_file, dwi_mask_file, 
                        denoised_dwi_file, residual_dwi_file=None, force_flg=False):
    '''
    Smoothing the dwi using the dipy patch2self (not very good)    
    '''
    if not exists(denoised_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=100)
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(dwi_mask_file).get_fdata()
        dwi = applymask(dwi, mask)
        
        # smoothing (change ols to rigid to reduce computation cost)
        denoised_dwi = patch2self(dwi, bvals, model='ols', shift_intensity=True,
                                  clip_negative_vals=False, b0_threshold=100)
        denoised_dwi = applymask(denoised_dwi, mask)
        save_nifti(denoised_dwi_file, denoised_dwi, dwi_nii.affine, dwi_nii.header)

        if residual_dwi_file is not None:
            residual_dwi_data = dwi - denoised_dwi
            save_nifti(residual_dwi_file, residual_dwi_data, dwi_nii.affine, dwi_nii.header)


def grad2fslgrad(bval_file, bvec_file, save_fslgrad_file, b0_thr=80):

    '''
    Convert bval_file and bvec_file to gradient table and save as save_fslgrad_file
    
    '''
    assert not exists(save_fslgrad_file)
    
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)
    
    fslgrad = np.concatenate([gtab.bvecs, gtab.bvals[:,np.newaxis]], axis=1)
    np.savetxt(save_fslgrad_file, fslgrad, fmt='%.5f', delimiter=' ')


def fslgrad2grad(fslgrad_file, save_bval_file, save_bvec_file):

    '''
    Convert fslgrad_file to bval and bvec
    '''
    assert not exists(save_bval_file)
    assert not exists(save_bvec_file)
    
    gt_df = pd.read_table(fslgrad_file, sep=' ', names=['x','y','z','b'])

    with open(save_bval_file,"w") as f:
        f.write(" ".join(str(int(x)) for x in list(gt_df['b'])))
        f.write("\n")

    with open(save_bvec_file,"w") as f:
        dim_str_list = []
        for dim in ['x','y','z']:
            dim_str = " ".join("{:.6f}".format(x) for x in list(gt_df[dim].astype(float)))
            dim_str_list.append(dim_str)
        f.write("\n".join(x for x in dim_str_list))
        f.write("\n")
        

def extract_single_shell(dwi_file, bval_file, bvec_file, 
                         splited_dwi_file, splited_bval_file, splited_bvec_file, 
                         bval_lthr, bval_uthr, b0_thr=80, force_flg=False, return_flg=False):
    '''
    Extract the single shell from multi-shell dwi data   
    '''
    if not exists(splited_dwi_file) or force_flg:
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        
        # assert np.any(((bvals <= bval_uthr) & (bvals > bval_lthr))), 'Shell not detected, check bval!'
        
        if np.any(((bvals <= bval_uthr) & (bvals > bval_lthr))):

            gradmask = (bvals < b0_thr) | ((bvals <= bval_uthr) & (bvals > bval_lthr))
            split_bvals = bvals[gradmask]
            split_bvecs = bvecs[gradmask,:]
            split_dwi = dwi[...,gradmask]
            
            save_nifti(splited_dwi_file, split_dwi, dwi_nii.affine, dwi_nii.header)
            np.savetxt(splited_bval_file, split_bvals[:,np.newaxis].T, fmt='%d', delimiter=' ')
            np.savetxt(splited_bvec_file, split_bvecs.T, fmt='%.5f', delimiter=' ')      
            
            if return_flg:
                return split_dwi, split_bvals, split_bvecs
        
        else:
            print(f'No shell in target bval range [{bval_uthr}, {bval_lthr}]!')

