'''
Calculate dMRI features:  DTI, RISH, Moment

'''
from utils import *
from diff_utils import *

def gen_dti(dwi_file, mask_file,  bval_file, bvec_file, 
            fa_file, md_file, gfa_file=None, 
            b0_thr=80, force_flg=False):
    '''
    Generate DTI images: FA, MD. GFA


    '''
    if not (exists(fa_file) and exists(md_file)) or force_flg:
        # 1. load dwi data
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        dwi_affine, dwi_header = dwi_nii.affine, dwi_nii.header

        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

        # 2. calculate FA and MD
        dtimodel = reconst_dti.TensorModel(gtab, fit_method ="LS")
        dtifit = dtimodel.fit(dwi)
        save_nifti(fa_file, dtifit.fa, dwi_nii.affine, dwi_nii.header)
        save_nifti(md_file, dtifit.md, dwi_nii.affine, dwi_nii.header)

    # 3. calculate GFA
    if gfa_file is not None:
        if not exists(gfa_file) or force_flg:
            # 1. in original rish harmonization not correct
            # gfa_vol = np.nan_to_num(gfa(dwi))
            # save_nifti(gfa_file, gfa_vol, dwi_nii.affine, dwi_nii.header)

            # 2. Using Shore for ODF and gfa (slow)
            # from dipy.reconst.shore import ShoreModel
            # from dipy.data import get_fnames, get_sphere
            # radial_order = 6
            # zeta = 700
            # lambdaN = 1e-8
            # lambdaL = 1e-8
            # asm = ShoreModel(gtab, radial_order=radial_order,
            #                  zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
            # asmfit = asm.fit(dwi)
            # sphere = get_sphere('repulsion724')
            # odf = asmfit.odf(sphere)
            # gfa_vol = np.nan_to_num(gfa(dwi_norm))
            # save_nifti(gfa_file, gfa_vol, dwi_nii.affine, dwi_nii.header)

            # 3. Using csaODFModel
            from dipy.data import default_sphere
            from dipy.reconst.shm import CsaOdfModel
            from dipy.direction import peaks_from_model
            csamodel = CsaOdfModel(gtab, 6)
            csapeaks = peaks_from_model(model=csamodel,
                                data=dwi,
                                sphere=default_sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True)

            gfa_vol = csapeaks.gfa
            save_nifti(gfa_file, gfa_vol, dwi_nii.affine, dwi_nii.header)

def gen_dti2(dwi_file, mask_file,  bval_file, bvec_file, 
            featfiles, featnames,
            b0_thr=80, force_flg=False):
    '''
    Generate DTI images: FA, Mean Diffusivity (MD), Axial Diffusivity (AD) , Radial Diffusivity (RD), Geodesic anisotropy (GA)
    '''

    print(featfiles)
    print(np.all([exists(dti_file) for dti_file in featfiles]))

    if (not np.all([exists(featfile) for featfile in featfiles])) or force_flg:
        # 1. load dwi data
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        dwi_affine, dwi_header = dwi_nii.affine, dwi_nii.header

        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

        # 2. calculate FA and MD
        dtimodel = reconst_dti.TensorModel(gtab, fit_method ="LS")
        dtifit = dtimodel.fit(dwi)

        for featname, featfile in zip(featnames, featfiles):
            # print(featname, featfile)
            save_nifti(featfile, getattr(dtifit, featname), dwi_nii.affine, dwi_nii.header)


def gen_dki(dwi_file, mask_file,  bval_file, bvec_file, 
            featfiles, featnames, fit_method='OLS', b0_thr=80, force_flg=False):
    '''
    A. Generate DKI images
    FA, Mean Diffusivity (MD), Axial Diffusivity (AD) , Radial Diffusivity (RD), mean kurtosis (MK), the axial kurtosis (AK) and the radial kurtosis (RK)
    
    B. fit_method : str or callable, optional
        str be one of the following:
            'OLS' or 'ULLS' for ordinary least squares.
            'WLS', 'WLLS' or 'UWLLS' for weighted ordinary least squares.
                See dki.ls_fit_dki.
            'CLS' for LMI constrained ordinary least squares [2].
            'CWLS' for LMI constrained weighted least squares [2].
                See dki.cls_fit_dki.

    '''

    print(featfiles)
    print(np.all([exists(featfile) for featfile in featfiles]))

    if (not np.all([exists(featfile) for featfile in featfiles])) or force_flg:
        # 1. load dwi data
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        dwi_affine, dwi_header = dwi_nii.affine, dwi_nii.header

        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

        # 2. calculate FA and MD
        dkimodel = reconst_dki.DiffusionKurtosisModel(gtab, fit_method=fit_method)
        dkifit = dkimodel.fit(dwi, mask=mask)

        for featname, featfile in zip(featnames, featfiles):
            print(featname, featfile)

            feat_data = getattr(dkifit, featname)
            if hasattr(feat_data, '__call__'):
                feat_data = feat_data()

            print(feat_data.shape)
            
            save_nifti(featfile, feat_data, dwi_nii.affine, dwi_nii.header)


def gen_msdki(dwi_file, mask_file,  bval_file, bvec_file, 
            featfiles, featnames, b0_thr=80, force_flg=False):

    '''
    mean signal diffusion kurtosis imaging (MSDKI)

    1) the mean signal diffusion (MSD); and 2) the mean signal kurtosis (MSK)
    '''
    
    print(featfiles)
    print(np.all([exists(featfile) for featfile in featfiles]))

    if (not np.all([exists(featfile) for featfile in featfiles])) or force_flg:
        # 1. load dwi data
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        dwi_affine, dwi_header = dwi_nii.affine, dwi_nii.header

        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

        # 2. calculate FA and MD
        msdki_model = reconst_msdki.MeanDiffusionKurtosisModel(gtab)
        msdki_fit = msdki_model.fit(dwi, mask=mask)

        for featname, featfile in zip(featnames, featfiles):
            print(featname, featfile)

            feat_data = getattr(msdki_fit, featname)
            if hasattr(feat_data, '__call__'):
                feat_data = feat_data()

            save_nifti(featfile, feat_data, dwi_nii.affine, dwi_nii.header)


        
def gen_rish(dwi_file, mask_file, bval_file, bvec_file,
            rish_file_pattern, sh_num, 
            b0_thr=80, force_flg=False):
    
    '''
    Generate RISH images
    
    '''
    
    # 1. Check if expected rish files exists  
    rish_exist_flg = len(glob(rish_file_pattern)) >= (sh_num//2 + 1)
    
    # 2. calculate RISH features
    if not rish_exist_flg or force_flg:
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        
        # form a full sphere
        bvals = np.append(bvals, bvals)
        bvecs = np.append(bvecs, -bvecs, axis=0)
        dwi = np.append(dwi, dwi, axis=3)
        
#             dwi_norm, _ = normalize_data(dwi, where_b0=np.where(qb_model.gtab.b0s_mask)[0])
        dwi_norm, _ = normalize_data(dwi, where_b0=np.where(bvals < b0_thr)[0], mask=mask)
        
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)
        qb_model = QballModel(gtab, sh_order=sh_num)
        
        # inserting correct shm_coeff computation block ---------------------------------
        smooth = 0.00001

        L = qb_model.n*(qb_model.n+1)
        L**=2
        _fit_matrix = np.linalg.pinv(qb_model.B.T @ qb_model.B + np.diag(smooth*L)) @ qb_model.B.T
        shm_coeff = np.dot(dwi_norm[..., qb_model._where_dwi], _fit_matrix.T)
        shm_coeff = applymask(shm_coeff, mask)
        # -------------------------------------------------------------------------------

        shm_coeff_squared = shm_coeff**2
        shs_same_level = [[0, 1], [1, 6], [6, 15], [15, 28], [28, 45]]
        
        shs_files = []
        for i in range(0, sh_num+1, 2):
            ind = int(i/2)
            temp = np.sum(shm_coeff_squared[:,:,:,shs_same_level[ind][0]:shs_same_level[ind][1]], axis=3)
            shs_file = rish_file_pattern.replace('*',f'{ind*2}')
            shs_files.append(shs_file)
            save_nifti(shs_file, temp, dwi_nii.affine, dwi_nii.header)
    else:
        shs_files = []
        for i in range(0, sh_num+1, 2):
            ind = int(i/2)
            shs_file = rish_file_pattern.replace('*',f'{ind*2}')
            assert exists(shs_file)
            shs_files.append(shs_file)
        
    return shs_files

   
def gen_moment(dwi_file, mask_file, bval_file, bvec_file,
            moment_file_pattern, 
            b0_thr=80, force_flg=False):
    
    '''
    Generate dMRI Moment images (first order moment and Second Centrial Moment)
    
    '''
    
    # 1. Check if expected rish files exists  
    moment_files = sorted(glob(moment_file_pattern))
    moment_exist_flg = len(moment_files) == 2
    
    # 2. calculate Moments
    if not moment_exist_flg or force_flg:
        dwi_nii = load(dwi_file)
        dwi = dwi_nii.get_fdata()
        mask = load(mask_file).get_fdata() > 0
        dwi = applymask(dwi, mask)
        
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        
        # form a full sphere
        bvals = np.append(bvals, bvals)
        bvecs = np.append(bvecs, -bvecs, axis=0)
        dwi = np.append(dwi, dwi, axis=3)
        
        # normalized dMRI
        dwi_norm, _ = normalize_data(dwi, where_b0=np.where(bvals < b0_thr)[0], mask=mask)
        dwi_norm_b1 = dwi_norm[...,bvals > b0_thr] # extract the non b0 diffusion signal

        # estimate two moments
        org_moment1 = np.mean(dwi_norm_b1,axis=3)
        org_moment2 = np.sum((dwi_norm_b1 - org_moment1[...,None])**2, axis=3)/(dwi_norm_b1.shape[-1] - 1) 
        
        moment1_file = moment_file_pattern.replace('*', str(1))
        moment2_file = moment_file_pattern.replace('*', str(2))
        
        save_nifti(moment1_file, org_moment1, dwi_nii.affine, dwi_nii.header)
        save_nifti(moment2_file, org_moment2, dwi_nii.affine, dwi_nii.header)
        
        moment_files = [moment1_file, moment2_file]
    
    return moment_files



def gen_noddi(dwi_file, mask_file, bval_file, bvec_file,
            odi_file, ndi_file, cpu_num=1, delta=0.0106, Delta=0.0431, b0_thr=80, force_flg=False):
    

    from dipy.core.gradients import gradient_table
    from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
    from dmipy.signal_models import cylinder_models, gaussian_models
    from dmipy.distributions.distribute_models import SD1WatsonDistributed
    from dmipy.core.modeling_framework import MultiCompartmentModel


    if not exists(odi_file) or not exists(odi_file) or force_flg:

        assert exists(dwi_file), f'{dwi_file} does not exists!'

        dwi_img = load(dwi_file).get_fdata()
        dwi_affine = load(dwi_file).affine
        dwi_header = load(dwi_file).header
        mask_img = load(mask_file).get_fdata() > 0

        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_thr, big_delta=Delta, small_delta=delta)
        acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_thr*1e6)
        acq_scheme_mipy.print_acquisition_info

        # models
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()


        watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
        watson_dispersed_bundle.parameter_names

        watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)


        NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
        NODDI_mod.parameter_names
        NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

        odi_box = []
        ndi_box = []
        dwi_box = []

        for i in range(dwi_img.shape[1]):
            print(f'NODDI for x_slice: {i}/{dwi_img.shape[1]}')
            # print(dwi_img[:,i:i+1,...].shape)
            # print(mask_img[:,i:i+1,...].shape)

            if np.any(mask_img[:,i:i+1,...]):
            
                NODDI_fit_hcp = NODDI_mod.fit(acq_scheme_mipy, 
                                dwi_img[:,i:i+1,...], mask=mask_img[:,i:i+1,...]>0, number_of_processors=cpu_num)

                # # get odi
                odi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
                # # get ndi
                ndi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']

                # # get total Stick signal contribution
                # vf_intra = (NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
                #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

                # # get total Zeppelin signal contribution
                # vf_extra = ((1 - NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
                #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])
            else:
                odi = np.zeros(mask_img[:,i:i+1,...].shape)
                ndi = np.zeros(mask_img[:,i:i+1,...].shape)

            odi_box.append(odi)
            ndi_box.append(ndi)

        odi = np.concatenate(odi_box, axis=1)
        ndi = np.concatenate(ndi_box, axis=1)

        save_nifti(odi_file, odi, dwi_affine, hdr=dwi_header)
        save_nifti(ndi_file, ndi, dwi_affine, hdr=dwi_header)