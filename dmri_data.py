
from utils import *
from diff_utils import *

from dmri_feature import gen_dti, gen_rish, gen_moment, gen_dti2, gen_dki, gen_msdki
from dmri_preprocessing import denoise_dwi_mppca, denoise_dwi_native_mppca, denoise_dwi_localpca, denoise_dwi_patch2self, denoise_dwi_all, denoise_dwi_gibbs, extract_single_shell
from ants_tools import mri_registration, recent_nii, derecent_nii

class T1_data():
    '''
    T1 data class
    '''

    def __init__(self, T1_file, Aseg_file=None, sub_name=None, site_name=None, sub_dir=None):
        self.T1_file = T1_file
        self.Aseg_file = Aseg_file
        self.sub_name = sub_name
        self.site_name = site_name
        self.sub_dir = sub_dir


class demo_data():

    def __init__(self, sub_name, age=None, gender=None):
        self.sub_name = sub_name
        self.age = age
        self.gender = gender


class diffusion_data():
    '''
    dMRI data class
    '''
    def __init__(self, dwi_file, bvec_file, bval_file, dwi_mask_file, sub_name=None, site_name=None, sub_dir=None):
        
        # file info.
        self.dwi_file = dwi_file
        self.bvec_file = bvec_file
        self.bval_file = bval_file
        self.mask_file = dwi_mask_file

        assert exists(dwi_file), f'{dwi_file} does not exists!'
        assert exists(bvec_file), f'{bvec_file} does not exists!'
        assert exists(bval_file), f'{bval_file} does not exists!'
        assert exists(dwi_mask_file), f'{dwi_mask_file} does not exists!'

        # subject info.
        self.sub_name = sub_name
        self.site_name = site_name

        if sub_dir is None:
            self.sub_dir = dirname(self.dwi_file)
        else:
            self.sub_dir = sub_dir

        
    # name of the diffusion feature should be 'sub_dir + feature_sub_dir + sub_name + feature_postfix' 
    def make_dti(self, dti_dir='dti', b0_thr=100, force_flg=False, nogfa=True):
        
        # make dti_dir
        sub_dti_dir = pjoin(self.sub_dir, dti_dir)
        create_dir(sub_dti_dir)
        
        # DTI filename
        if self.sub_name is None:
            self.fa_file = pjoin(sub_dti_dir, f'FA.nii.gz')
            self.md_file = pjoin(sub_dti_dir, f'MD.nii.gz')
            self.gfa_file = pjoin(sub_dti_dir, f'GFA.nii.gz')
        else:
            self.fa_file = pjoin(sub_dti_dir, f'{self.sub_name}_FA.nii.gz')
            self.md_file = pjoin(sub_dti_dir, f'{self.sub_name}_MD.nii.gz')
            self.gfa_file = pjoin(sub_dti_dir, f'{self.sub_name}_GFA.nii.gz')
        
        if nogfa:
            self.gfa_file = None # do not need gfa

        # calculate DTI
        gen_dti(self.dwi_file, self.mask_file,  self.bval_file, self.bvec_file, 
            self.fa_file, self.md_file, gfa_file=self.gfa_file, 
            b0_thr=b0_thr, force_flg=force_flg)

    # name of the diffusion feature should be 'sub_dir + feature_sub_dir + sub_name + feature_postfix' 
    def make_dti2(self, dti_dir='dti2', b0_thr=100, force_flg=False):
        
        # make dti_dir
        sub_dti_dir = pjoin(self.sub_dir, dti_dir)
        create_dir(sub_dti_dir)

        # maps = ['fa', 'md', 'ad', 'rd', 'mk', 'ak', 'rk']   mean kurtosis (MK), the axial kurtosis (AK) and the radial kurtosis (RK)
        # FA, Mean Diffusivity (MD), Axial Diffusivity (AD) , Radial Diffusivity (RD), Geodesic anisotropy (GA)
        featnames = ['fa', 'md', 'ad', 'rd', 'ga']
        featfiles = []

        for featname in featnames:
            if self.sub_name is None:
                featfile = pjoin(sub_dti_dir, f'{featname}.nii.gz')
            else:
                featfile = pjoin(sub_dti_dir, f'{self.sub_name}_{featname}.nii.gz')

            setattr(self, f'{featname}_file', featfile)
            featfiles.append(featfile)
        
        # calculate DTI
        gen_dti2(self.dwi_file, self.mask_file,  self.bval_file, self.bvec_file, 
            featfiles, featnames, b0_thr=b0_thr, force_flg=force_flg)

    def make_dki(self, feat_dir='dki', fit_method='OLS', b0_thr=100, force_flg=False):
        
        # make sub_feat_dir
        sub_feat_dir = pjoin(self.sub_dir, feat_dir)
        create_dir(sub_feat_dir)

        # FA, Mean Diffusivity (MD), Axial Diffusivity (AD) , Radial Diffusivity (RD), mean kurtosis (MK), the axial kurtosis (AK) and the radial kurtosis (RK)
        featnames = ['fa', 'md', 'ad', 'rd', 'mk', 'ak', 'rk']   
        featfiles = []

        for featname in featnames:
            if self.sub_name is None:
                featfile = pjoin(sub_feat_dir, f'{featname}.nii.gz')
            else:
                featfile = pjoin(sub_feat_dir, f'{self.sub_name}_{featname}.nii.gz')

            setattr(self, f'{featname}_file', featfile)
            featfiles.append(featfile)
        
        # calculate DTI
        gen_dki(self.dwi_file, self.mask_file, self.bval_file, self.bvec_file, 
            featfiles, featnames, fit_method=fit_method, b0_thr=b0_thr, force_flg=force_flg)

    def make_msdki(self, feat_dir='msdki', b0_thr=100, force_flg=False):
        
        # make sub_feat_dir
        sub_feat_dir = pjoin(self.sub_dir, feat_dir)
        create_dir(sub_feat_dir)

        # 1) the mean signal diffusion (MSD); and 2) the mean signal kurtosis (MSK)
        featnames = ['msd', 'msk']   
        featfiles = []

        for featname in featnames:
            if self.sub_name is None:
                featfile = pjoin(sub_feat_dir, f'{featname}.nii.gz')
            else:
                featfile = pjoin(sub_feat_dir, f'{self.sub_name}_{featname}.nii.gz')

            setattr(self, f'{featname}_file', featfile)
            featfiles.append(featfile)
        
        # calculate DTI
        gen_msdki(self.dwi_file, self.mask_file, self.bval_file, self.bvec_file, 
            featfiles, featnames, b0_thr=b0_thr, force_flg=force_flg)
  
    def make_rish(self, rish_dir='rish', sh_num=8, b0_thr=100, force_flg=False):
        
        # make sub_rish_dir
        sub_rish_dir = pjoin(self.sub_dir, rish_dir)
        create_dir(sub_rish_dir)
        
        # RISH filename pattern
        if self.sub_name is None:
            rish_file_pattern = pjoin(sub_rish_dir, f'rish_l*.nii.gz')
        else:
            rish_file_pattern = pjoin(sub_rish_dir, f'{self.sub_name}_rish_l*.nii.gz')


        # calculate RISH 
        self.shs_files = gen_rish(self.dwi_file, self.mask_file, self.bval_file, self.bvec_file,
                        rish_file_pattern, sh_num, 
                        b0_thr=b0_thr, force_flg=force_flg)
        
        
    def make_moment(self, moment_dir='moment', b0_thr=100, force_flg=False):
    
        # make sub_rish_dir
        sub_moment_dir = pjoin(self.sub_dir, moment_dir)
        create_dir(sub_moment_dir)
        
        # Moment filename pattern
        if self.sub_name is None:
            moment_file_pattern = pjoin(sub_moment_dir, f'moment*.nii.gz')
        else:
            moment_file_pattern = pjoin(sub_moment_dir, f'{self.sub_name}_moment*.nii.gz')

        self.moment_files = gen_moment(self.dwi_file, self.mask_file, self.bval_file, self.bvec_file,
                            moment_file_pattern, 
                            b0_thr=b0_thr, force_flg=force_flg)


    def make_noddi(self, noddi_dir='noddi', b0_thr=100, cpu_num=1, force_flg=False):

        from dmri_feature import gen_noddi
        print(f'Using {cpu_num} cpus for noddi reconstruction!')

        # make sub_rish_dir
        sub_noddi_dir = pjoin(self.sub_dir, noddi_dir)
        create_dir(sub_noddi_dir)
        
        # Moment filename pattern
        # Moment filename pattern
        if self.sub_name is None:
            self.odi_file = pjoin(sub_noddi_dir, f'ODI.nii.gz')
            self.ndi_file = pjoin(sub_noddi_dir, f'NDI.nii.gz')
        else:
            self.odi_file = pjoin(sub_noddi_dir, f'{self.sub_name}_ODI.nii.gz')
            self.ndi_file = pjoin(sub_noddi_dir, f'{self.sub_name}_NDI.nii.gz')


        gen_noddi(self.dwi_file, self.mask_file, self.bval_file, self.bvec_file, self.odi_file, self.ndi_file, 
            cpu_num=cpu_num, delta=0.0106, Delta=0.0431, b0_thr=b0_thr, force_flg=force_flg)



    # preprocessing
    def denoise(self, denoised_dwi_file, deno_mode='lp', residual_dwi_file=None, force_flg=False):
        '''
        denoise dwi using local pca
        '''
        if deno_mode == 'lp':
            denoise_dwi_localpca(self.dwi_file, self.bval_file, self.bvec_file, self.mask_file,
                denoised_dwi_file, residual_dwi_file=residual_dwi_file, force_flg=force_flg)
        elif deno_mode == 'mp':
            denoise_dwi_mppca(self.dwi_file, self.bval_file, self.bvec_file, self.mask_file, 
                        denoised_dwi_file, residual_dwi_file=residual_dwi_file, force_flg=force_flg)
        else:
            denoise_dwi_native_mppca(self.dwi_file, self.bval_file, self.bvec_file, self.mask_file, 
                        denoised_dwi_file, residual_dwi_file=residual_dwi_file, force_flg=force_flg)

        self.dwi_file = denoised_dwi_file


    def gibbs_correct(self, denoised_dwi_file, residual_dwi_file=None, force_flg=False):
        '''
        denoise dwi using gibbs removal
        '''
        denoise_dwi_gibbs(self.dwi_file, self.bval_file, self.bvec_file, self.mask_file,
            denoised_dwi_file, residual_dwi_file=residual_dwi_file, force_flg=force_flg)

        self.dwi_file = denoised_dwi_file


    def denoise_all(self, denoised_dwi_file, residual_dwi_file=None, force_flg=False):
        '''
        denoise dwi using gibbs removal + denoising using local pca
        '''
        denoise_dwi_all(self.dwi_file, self.bval_file, self.bvec_file, self.mask_file,
            denoised_dwi_file, residual_dwi_file=residual_dwi_file, force_flg=force_flg)

        self.dwi_file = denoised_dwi_file



    # extract single shell
    def extract_shell(self, splited_dwi_file, splited_bval_file, splited_bvec_file, splited_mask_file, 
                         shell_bval, b0_thr=100, force_flg=False):

        bval_lthr = shell_bval-400
        bval_uthr = shell_bval+400

        extract_single_shell(self.dwi_file, self.bval_file, self.bvec_file, 
                         splited_dwi_file, splited_bval_file, splited_bvec_file, 
                         bval_lthr, bval_uthr, b0_thr=b0_thr, force_flg=force_flg, return_flg=False)

        if not exists(splited_mask_file): 
            shutil.copy(self.mask_file, splited_mask_file)

        return diffusion_data(splited_dwi_file, splited_bvec_file, splited_bval_file, splited_mask_file, 
                    sub_name=self.sub_name, site_name=self.site_name, sub_dir=dirname(splited_dwi_file))


    # warpping
    def template_warp(self, template_dir, template_img):
        '''
        Load the precalculated transform staff
        '''
        assert(self.sub_name is not None)

        warp_field = None
        inverse_warp_field = None
        affine = None

        if glob(f'{template_dir}/T_{self.sub_name}*[0-9]Warp.nii.gz'):
            warp_field = glob(f'{template_dir}/T_{self.sub_name}*[0-9]Warp.nii.gz')[0]
            inverse_warp_field = glob(f'{template_dir}/T_{self.sub_name}*[0-9]InverseWarp.nii.gz')[0]
            affine = glob(f'{template_dir}/T_{self.sub_name}*[0-9]GenericAffine.mat')[0]

        elif glob(f'{template_dir}/Warp.nii.gz'):
            warp_field = f'{template_dir}/Warp.nii.gz'
            inverse_warp_field = f'{template_dir}/InverseWarp.nii.gz'
            affine = f'{template_dir}/Affine.txt'

            print(warp_field, exists(warp_field))

        self.regi = mri_registration(warp_field, inverse_warp_field, affine, template_img, sub_name=self.sub_name)


    def forward_warp(self, moving_img, output_img, recent_flg=False, nn_flg=False, force_flg=False):
        # print(recent_flg)
        create_dir(dirname(output_img))

        if recent_flg:
            # print('before', moving_img)
            moving_img = recent_nii(moving_img)
            # print('after', moving_img)

        self.regi.temp_forward(moving_img, output_img, self.regi.template_img, nn_flg=nn_flg, force_flg=force_flg)


    def backward_warp(self, reference_img, output_img, moving_img, recent_flg=False, nn_flg=False, force_flg=False):

        create_dir(dirname(output_img))

        if recent_flg:
            recent_moving_img = recent_nii(moving_img)

            self.regi.temp_backward(reference_img, output_img, recent_moving_img, nn_flg=nn_flg, force_flg=force_flg)

            # decent the output image
            moving_nii = load(moving_img)
            save_nifti(output_img, load(output_img).get_fdata(), moving_nii.affine, moving_nii.header)

        else:
            self.regi.temp_backward(reference_img, output_img, moving_img, nn_flg=nn_flg, force_flg=force_flg)


    def temp_forward_dti(self, template_name='temp', recent_flg=False):

        assert(exists(self.fa_file))

        self.fa_temp_file = add_fname_postfix(self.fa_file, template_name)
        self.forward_warp(self.fa_file, self.fa_temp_file, recent_flg=recent_flg)

        self.md_temp_file = add_fname_postfix(self.md_file, template_name)
        self.forward_warp(self.md_file, self.md_temp_file, recent_flg=recent_flg)

        # if self.gfa_file is not None:
        #     self.gfa_temp_file = add_fname_postfix(self.gfa_file, template_name)
        #     self.forward_warp(self.gfa_file, self.gfa_temp_file, recent_flg=recent_flg)

        featnames = ['fa', 'md', 'ad', 'rd', 'ga']
        for featname in featnames:
            setattr(self, f'{featname}_temp_file', add_fname_postfix(getattr(self, f'{featname}_file'), template_name))
            self.forward_warp(getattr(self, f'{featname}_file'), getattr(self, f'{featname}_temp_file'), recent_flg=recent_flg)

            print(getattr(self, f'{featname}_file'), exists(getattr(self, f'{featname}_file')))
            print(getattr(self, f'{featname}_temp_file'), exists(getattr(self, f'{featname}_temp_file')))
            
    def temp_forward_dwi(self, template_name='temp', recent_flg=False, force_flg=False):
        '''
        Warp the dwi to template space

        '''

        assert(exists(self.dwi_file))
        assert(exists(self.fa_file))
        assert(exists(self.fa_temp_file))
        

        self.dwi_temp_file = add_fname_postfix(self.dwi_file, 'temp')

        if not exists(self.dwi_temp_file) or force_flg:

            # split dwi to slices and warp
            self.dwi_temp_slice_files = []
            dwi_slice_file = add_fname_postfix(self.dwi_file, 'slice')
            dwi_img = load(self.dwi_file).get_fdata()
            for i in range(dwi_img.shape[-1]):
                dwi_slice = dwi_img[..., i]
                adwi_slice_file = dwi_slice_file.replace('.nii.gz',f'_{i}.nii.gz')
                save_nifti(adwi_slice_file, dwi_slice, ref_fname=self.fa_file)

                dwi_slice_temp_file = add_fname_postfix(adwi_slice_file, template_name)
                self.forward_warp(adwi_slice_file, dwi_slice_temp_file, recent_flg=recent_flg, force_flg=force_flg)
                self.dwi_temp_slice_files.append(dwi_slice_temp_file)
            
            # merge warped slices
            dwi_temp_img = []
            for file in self.dwi_temp_slice_files:
                dwi_temp_img.append(np.expand_dims(load(file).get_fdata(), axis=3))
            dwi_temp_img = np.concatenate(dwi_temp_img, axis=3)
            
            save_nifti(self.dwi_temp_file, dwi_temp_img, ref_fname=self.fa_temp_file)
        
        slice_folder = pjoin(dirname(self.dwi_file), 'slice')
        if exists(slice_folder):
            print(f'Removing slice folder {slice_folder}')
            shutil.rmtree(slice_folder)

    def temp_forward_dwilike(self, dwi_like_file, dwi_like_temp_file, template_name='temp', recent_flg=False, force_flg=False):
        '''
        Warp the dwi to template space

        '''

        assert(exists(dwi_like_file))
        assert(exists(self.fa_file))

        # self.dwi_temp_file = add_fname_postfix(dwi_like_file, 'temp')

        if not exists(dwi_like_temp_file) or force_flg:

            # split dwi to slices and warp
            dwi_like_temp_slice_files = []
            dwi_slice_file = add_fname_postfix(dwi_like_file, 'slice')
            dwi_img = load(dwi_like_file).get_fdata()
            for i in range(dwi_img.shape[-1]):
                dwi_slice = dwi_img[..., i]
                adwi_slice_file = dwi_slice_file.replace('.nii.gz',f'_{i}.nii.gz')
                save_nifti(adwi_slice_file, dwi_slice, ref_fname=self.fa_file)

                dwi_slice_temp_file = add_fname_postfix(adwi_slice_file, template_name)
                self.forward_warp(adwi_slice_file, dwi_slice_temp_file, recent_flg=recent_flg, force_flg=force_flg)
                dwi_like_temp_slice_files.append(dwi_slice_temp_file)
            
            # merge warped slices
            dwi_temp_img = []
            for file in dwi_like_temp_slice_files:
                dwi_temp_img.append(np.expand_dims(load(file).get_fdata(), axis=3))
            dwi_temp_img = np.concatenate(dwi_temp_img, axis=3)
            
            save_nifti(dwi_like_temp_file, dwi_temp_img, ref_fname=self.fa_file)
        
        slice_folder = pjoin(dirname(dwi_like_file), 'slice')
        if exists(slice_folder):
            print(f'Removing slice folder {slice_folder}')
            shutil.rmtree(slice_folder)


    def backward_dwi(self, dwi_temp_file, unwarp_dwi_file, untemp_post='', copy=False, remove=True):

        assert exists(dwi_temp_file)
        assert exists(self.fa_temp_file)
        assert exists(self.fa_file)

        self.dwi_untemp_slice_files = []
        dwi_slice_file = add_fname_postfix(dwi_temp_file, f'{untemp_post}slice')
        dwi_img = load(dwi_temp_file).get_fdata()
        for i in range(dwi_img.shape[-1]):
            dwi_slice = dwi_img[..., i]
            adwi_slice_file = dwi_slice_file.replace('.nii.gz',f'_{i}.nii.gz')
            if not exists(adwi_slice_file):
                save_nifti(adwi_slice_file, dwi_slice, ref_fname=self.fa_temp_file)

            dwi_slice_untemp_file = add_fname_postfix(adwi_slice_file, f'untemp')
            self.backward_warp(adwi_slice_file, dwi_slice_untemp_file, self.fa_file, recent_flg=True)    
            self.dwi_untemp_slice_files.append(dwi_slice_untemp_file)
            
        # unwarp_dwi_file = f'{dirname(self.dwi_file)}_{harm_ex}/{basename(self.dwi_file)}'

        # merge warped slices
        dwi_untemp_img = []
        for file in self.dwi_untemp_slice_files:
            dwi_untemp_img.append(np.expand_dims(load(file).get_fdata(), axis=3))
        dwi_untemp_img = np.concatenate(dwi_untemp_img, axis=3)

        save_nifti(unwarp_dwi_file, dwi_untemp_img, ref_fname=self.fa_file)

        if copy:
            copy2(self.mask_file, dirname(unwarp_dwi_file))
            copy2(self.bval_file, dirname(unwarp_dwi_file))
            copy2(self.bvec_file, dirname(unwarp_dwi_file))

        if remove:
            shutil.rmtree(dirname(dwi_slice_file))


    def temp_forward_gen(self, subspace_files, template_name='temp', recent_flg=False, nn_flg=False):
        commonspace_files = []
        for subspace_file in subspace_files:

            commonspace_file = add_fname_postfix(subspace_file, template_name)
            self.forward_warp(subspace_file, commonspace_file, recent_flg=recent_flg, nn_flg=nn_flg)
            commonspace_files.append(commonspace_file)
        return commonspace_files


    def temp_forward_rish(self, template_name='temp', recent_flg=False):
        if self.shs_files:
            self.shs_temp_files = []
            for shs_file in self.shs_files:

                shs_temp_file = add_fname_postfix(shs_file, template_name)
                self.forward_warp(shs_file, shs_temp_file, recent_flg=recent_flg)
                self.shs_temp_files.append(shs_temp_file)
                
    def temp_forward_moment(self, template_name='temp', recent_flg=False):
        if self.moment_files:
            self.moment_temp_files = []
            for moment_file in self.moment_files:
                moment_temp_file = add_fname_postfix(moment_file, template_name)
                self.forward_warp(moment_file, moment_temp_file, recent_flg=recent_flg)
                self.moment_temp_files.append(moment_temp_file)


    def load_seg_file(self, seg_file):
        self.seg_file = seg_file


def copy_dwi_data(diff_data, old_dir, new_dir, data_name=None):
    create_dir(new_dir)
    if data_name is not None:
        new_dwi_file = diff_data.dwi_file.replace(old_dir, new_dir)
        new_dwi_file = pjoin(dirname(new_dwi_file), f'{data_name}.nii.gz')
        new_mask_file = pjoin(dirname(new_dwi_file), f'{data_name}_mask.nii.gz')
        new_bval_file = new_dwi_file.replace('.nii.gz', '.bval')
        new_bvec_file = new_dwi_file.replace('.nii.gz', '.bvec')
    else:
        new_dwi_file = diff_data.dwi_file.replace(old_dir, new_dir)
        new_mask_file = diff_data.mask_file.replace(old_dir, new_dir)
        new_bval_file = new_dwi_file.replace('.nii.gz', '.bval')
        new_bvec_file = new_dwi_file.replace('.nii.gz', '.bvec')
    
    # create_dir
    create_dir(dirname(new_dwi_file))
    
    #copy
    if not exists(new_dwi_file): 
        shutil.copy(diff_data.dwi_file, new_dwi_file)
    if not exists(new_mask_file): 
        shutil.copy(diff_data.mask_file, new_mask_file)
    if not exists(new_bval_file): 
        shutil.copy(diff_data.bval_file, new_bval_file)
    if not exists(new_bvec_file): 
        shutil.copy(diff_data.bvec_file, new_bvec_file)
    
    print('*')
    print('Moving', diff_data.dwi_file, '\n->', new_dwi_file)
    print('Moving', diff_data.mask_file, '\n->', new_mask_file)
    print('Moving', diff_data.bval_file, '\n->', new_bval_file)
    print('Moving', diff_data.bvec_file, '\n->', new_bvec_file)

    new_diff_data = diffusion_data(new_dwi_file, new_bvec_file, new_bval_file, new_mask_file, sub_name=diff_data.sub_name, site_name=diff_data.site_name)
    
    
    # # validate
    # dwi_img = load(diff_data.dwi_file).get_fdata()
    # new_dwi_img = load(new_diff_data.dwi_file).get_fdata()    
    # assert (dwi_img == new_dwi_img).all()
    
    # mask_img = load(diff_data.mask_file).get_fdata() > 0
    # new_mask_img = load(new_diff_data.mask_file).get_fdata() > 0    
    # assert (mask_img == new_mask_img).all()
    
    # bvals, bvecs = read_bvals_bvecs(diff_data.bval_file, diff_data.bvec_file)
    # new_bvals, new_bvecs = read_bvals_bvecs(new_diff_data.bval_file, new_diff_data.bvec_file)
    # assert (new_bvals == bvals).all()
    # assert (new_bvecs == bvecs).all()
    
    return new_diff_data


class structure_data():
    '''
    structure data class
        a. aseg_file should contain the brain segmentation and parcellation info.
    '''
    def __init__(self, struct_file, aseg_file=None, sub_name=None, site_name=None, sub_dir=None):
        self.struct_file = struct_file
        self.aseg_file = aseg_file
