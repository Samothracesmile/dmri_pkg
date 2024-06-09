from utils import *
from diff_utils import *
from scipy import stats
from dipy.align.imaffine import AffineMap


def cal_qry_ns_qry_s_corr(qry_anat_img_file, template_mask_img, local_shifts, ddof):

    qry_non_shift_qry_shift_corr_box = []    
    qry_anat_img_data, qry_anat_img_affine = load_nifti(qry_anat_img_file)
    nonshift_qry_anat_img_fpatch = extract_fpatch(qry_anat_img_data, template_mask_img, w_size=1, ddof=ddof)
    for shift_idx, xyz_shift in enumerate(local_shifts):
        # qry non-shift vs qry shift
        shift_qry_anat_img_data = gen_shift_image(qry_anat_img_file, xyz_shift, shift_img_file=None)
        shift_qry_anat_img_fpatch = extract_fpatch(shift_qry_anat_img_data, template_mask_img, w_size=1, ddof=ddof)
        qry_ns_qry_s_patch_corr = cal_patchwise_corr(nonshift_qry_anat_img_fpatch, shift_qry_anat_img_fpatch)
        qry_non_shift_qry_shift_corr_box.append(qry_ns_qry_s_patch_corr)

    qry_non_shift_qry_shift_corr_box = np.array(qry_non_shift_qry_shift_corr_box).T
    return qry_non_shift_qry_shift_corr_box


def est_qry_ns_qry_s_corr_chunk(qry_anat_img_file, template_mask_img, local_shifts, ddof, chunk_num=1, chunk_idx=0, load_file=None):

    if (load_file is None) or (not exists(load_file)):
        np_array = cal_qry_ns_qry_s_corr(qry_anat_img_file, template_mask_img, local_shifts, ddof)
        if load_file is not None:
            save_npz(load_file, np_array)
    else:
        np_array = load_npz(load_file)
        
    if chunk_num <= 1:
        return np_array
    else:
        np_array_chunks = np.array_split(np_array, chunk_num)
        return np_array_chunks[chunk_idx]


def cal_qry_ns_ref_s_corr(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof):
    '''
    calculate qry center to ref neighborhood patch correlation
    '''

    qry_non_shift_ref_shift_corr_box = []    

    # load the qry anatomy image and extract patch
    qry_anat_img_data, qry_anat_img_affine = load_nifti(qry_anat_img_file)
    nonshift_qry_anat_img_fpatch = extract_fpatch(qry_anat_img_data, template_mask_img, w_size=1, ddof=ddof)

    for shift_idx, xyz_shift in enumerate(local_shifts):
        shift_ref_anat_img_data = gen_shift_image(ref_anat_img_file, xyz_shift, shift_img_file=None)
        shift_ref_anat_img_fpatch = extract_fpatch(shift_ref_anat_img_data, template_mask_img, w_size=1, ddof=ddof)
        qry_ns_ref_s_patch_corr = cal_patchwise_corr(nonshift_qry_anat_img_fpatch, shift_ref_anat_img_fpatch)
        qry_non_shift_ref_shift_corr_box.append(qry_ns_ref_s_patch_corr)

    qry_non_shift_ref_shift_corr_box = np.array(qry_non_shift_ref_shift_corr_box).T
    return qry_non_shift_ref_shift_corr_box


def est_qry_ns_ref_s_corr_chunk(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof, chunk_num=1, chunk_idx=0, load_file=None):

    if (load_file is None) or (not exists(load_file)):
        np_array = cal_qry_ns_ref_s_corr(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof)
        if load_file is not None:
            save_npz(load_file, np_array)
    else:
        np_array = load_npz(load_file)
        
    if chunk_num <= 1:
        return np_array
    else:
        np_array_chunks = np.array_split(np_array, chunk_num)
        return np_array_chunks[chunk_idx]    



def cal_qry_s_ref_s_corr(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof):
    '''
    calculate qry neighborhood to corresponding ref neighborhood patch correlation
    '''

    qry_shift_ref_shift_corr_box = []
    for shift_idx, xyz_shift in enumerate(local_shifts):

        # qry non-shift vs qry shift
        shift_qry_anat_img_data = gen_shift_image(qry_anat_img_file, xyz_shift, shift_img_file=None)
        shift_qry_anat_img_fpatch = extract_fpatch(shift_qry_anat_img_data, template_mask_img, w_size=1, ddof=ddof)

        # qry shift vs ref shift
        shift_ref_anat_img_data = gen_shift_image(ref_anat_img_file, xyz_shift, shift_img_file=None)
        shift_ref_anat_img_fpatch = extract_fpatch(shift_ref_anat_img_data, template_mask_img, w_size=1, ddof=ddof)
        qry_s_ref_s_patch_corr = cal_patchwise_corr(shift_qry_anat_img_fpatch, shift_ref_anat_img_fpatch)
        qry_shift_ref_shift_corr_box.append(qry_s_ref_s_patch_corr)

    qry_shift_ref_shift_corr_box = np.array(qry_shift_ref_shift_corr_box).T    
    return qry_shift_ref_shift_corr_box

def est_qry_s_ref_s_corr_chunk(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof, chunk_num=1, chunk_idx=0, load_file=None):

    if (load_file is None) or (not exists(load_file)):
        np_array = cal_qry_s_ref_s_corr(qry_anat_img_file, ref_anat_img_file, template_mask_img, local_shifts, ddof)
        if load_file is not None:
            save_npz(load_file, np_array)
    else:
        np_array = load_npz(load_file)
        
    if chunk_num <= 1:
        return np_array
    else:
        np_array_chunks = np.array_split(np_array, chunk_num)
        return np_array_chunks[chunk_idx]  


def extract_fpatch_old(anat_img, mask_img, w_size=1, ddof=1):
    
    '''
    Extract normalized patches from anat_img with mask_img:
    anat_img: 3D array
    mask_img: 3D array
    w_size: int: patch_size = 2*w_size + 1
    '''

    mask_indexs = np.array(np.where(mask_img)).T

    anat_patches = []
    for mask_index in mask_indexs:
        x_u, y_u, z_u = mask_index - w_size
        x_l, y_l, z_l = mask_index + w_size + 1
        anat_patches.append(anat_img[x_u:x_l, y_u:y_l, z_u:z_l].flatten())

    anat_patches = np.array(anat_patches)
    
    norm_anat_patches = stats.zscore(anat_patches, axis=1, ddof=ddof)

    return norm_anat_patches


def extract_fpatch(anat_img, mask_img, w_size=1, ddof=1, g_offset=[0,0,0]):
    
    '''
    Extract normalized patches from anat_img with mask_img:
    anat_img: 3D array
    mask_img: 3D array
    w_size: int: patch_size = 2*w_size + 1
    '''

    xg_offset, yg_offset, zg_offset = g_offset
    # print('g_offset', xg_offset, yg_offset, zg_offset)

    mask_indexs = np.array(np.where(mask_img)).T

    p_size = 2*w_size + 1
    xv,yv,zv = np.meshgrid(range(p_size),range(p_size),range(p_size))
    cubic_3_idx = np.array([xv.flatten()+xg_offset, yv.flatten()+yg_offset, zv.flatten()+zg_offset]).T - w_size

    anat_patches = []
    for (x_offset, y_offset, z_offset) in cubic_3_idx:
        anat_patches.append(anat_img[mask_indexs[:,0]+x_offset, mask_indexs[:,1]+y_offset, mask_indexs[:,2]+z_offset])
    anat_patches = np.array(anat_patches).T
    
    norm_anat_patches = stats.zscore(anat_patches, axis=1, ddof=ddof)
    
    return norm_anat_patches


def extract_epatch(anat_img, mask_img, w_size=1, g_offset=[0,0,0]):
    
    '''
    Extract normalized patches from anat_img with mask_img:
    anat_img: 3D array
    mask_img: 3D array
    w_size: int: patch_size = 2*w_size + 1
    '''

    xg_offset, yg_offset, zg_offset = g_offset
    # print('g_offset', xg_offset, yg_offset, zg_offset)

    mask_indexs = np.array(np.where(mask_img)).T

    p_size = 2*w_size + 1
    xv,yv,zv = np.meshgrid(range(p_size),range(p_size),range(p_size))
    cubic_3_idx = np.array([xv.flatten()+xg_offset, yv.flatten()+yg_offset, zv.flatten()+zg_offset]).T - w_size

    anat_patches = []
    for (x_offset, y_offset, z_offset) in cubic_3_idx:
        anat_patches.append(anat_img[mask_indexs[:,0]+x_offset, mask_indexs[:,1]+y_offset, mask_indexs[:,2]+z_offset])
    anat_patches = np.array(anat_patches).T
    
    norm_anat_patches = stats.zscore(anat_patches, axis=1, ddof=ddof)
    
    return norm_anat_patches

def cal_patchwise_corr(fpatch1, fpatch2, ddof=1):
    '''
    Calculation the correlaiton between patch arrays
    
    N: number of patch
    M: signal dimension
    
    fpatch1: z-scored patch array 1: (N,M)
    fpatch2: z-scored patch array 2: (N,M)
    
    ddof: degree of freedom
    
    Output: patchwise correlation (N,1)
    '''
    assert fpatch1.shape[-1] == fpatch2.shape[-1]
    vector_size = fpatch1.shape[-1]
    
    return np.sum(fpatch1*fpatch2, axis=1)/(vector_size - ddof)


def gen_shift_image(img_file, xyz_shift, shift_img_file=None):
    '''
    Generate shift image 
    
    img_file: nii file for shifting
    xyz_shift: offset of the shift [x_offset, y_offset, z_offset]
    Output: shift_img_file: nii file after shifting
    '''

    if (shift_img_file is not None) and exists(shift_img_file):
        # print(shift_img_file)

        shift_img_data, _ = load_nifti(shift_img_file)

        return shift_img_data

    else:

        img_data, img_affine = load_nifti(img_file)
        
        shift_img_affine = deepcopy(img_affine)
        shift_img_affine[:3,3] = np.array(xyz_shift)
        # print(shift_img_affine)
        
        affine_map = AffineMap(np.eye(4),
                           img_data.shape, img_affine,
                           img_data.shape, shift_img_affine) # direction is not important as we will apply this sysmetrically
        
        shift_img_data = affine_map.transform(img_data)
        
        if shift_img_file is not None:
            save_nifti(shift_img_file, shift_img_data, ref_fname=img_file)
            
        return shift_img_data
        