
from utils import *
from mask_utils import refine_brainmask
from dmri_data import diffusion_data


def load_diffdataset_from_csv(dwi_info_file):
    '''load diffdataset information from csv'''


    dwi_info_pd = pd.read_csv(dwi_info_file, sep="\t", header=0, index_col=0)

    diffdataset = []

    for idx, row in dwi_info_pd.iterrows():
        site_name = row['site_name']
        sub_id = row['sub_id']
        sub_dir = row['sub_dir']
        dwi_file = row['dwi_file']
        bval_file = row['bval_file']
        bvec_file = row['bvec_file']
        mask_file = row['mask_file']

        if isNaN(sub_id):
            sub_id = None
        if isNaN(sub_dir):
            sub_dir = None
        if isNaN(site_name):
            site_name = None
        
        diff_data = diffusion_data(dwi_file, bvec_file, bval_file, mask_file, 
                                        sub_name=sub_id, sub_dir=sub_dir, site_name=site_name)
    
        diffdataset.append(diff_data)

    print(30*'*')
    print(f'Loaded {len(diffdataset)} diffusion data.')
    return diffdataset


def save_diffdataset_to_csv(dwi_info_file, 
                            dwi_files, bval_files, bvec_files, mask_files, 
                            site_names=None, sub_ids=None, sub_dirs=None):


        '''save diffdataset information to csv'''

        if site_names is None:
            site_names = len(dwi_files) * [None]

        if sub_ids is None:
            sub_ids = len(dwi_files) * [None]

        if sub_dirs is None:
            sub_dirs = len(dwi_files) * [None]


        dwi_info_pd = pd.DataFrame({
                'site_name': site_names,   
                'sub_id': sub_ids,
                'sub_dir': sub_dirs,  
                'dwi_file': dwi_files,
                'bval_file': bval_files,
                'bvec_file': bvec_files,
                'mask_file': mask_files
            })

        dwi_info_pd.to_csv(dwi_info_file, sep="\t")


def cvt_diffdataset_to_csv(dwi_info_file, diffdataset):

    dwi_files = [diffdata.dwi_file for diffdata in diffdataset]
    bval_files = [diffdata.bval_file for diffdata in diffdataset]
    bvec_files = [diffdata.bvec_file for diffdata in diffdataset]
    mask_files = [diffdata.mask_file for diffdata in diffdataset]
    sub_ids = [diffdata.sub_name for diffdata in diffdataset]
    site_names = [diffdata.site_name for diffdata in diffdataset]
    sub_dirs = [diffdata.sub_dir for diffdata in diffdataset]

    save_diffdataset_to_csv(dwi_info_file, 
                            dwi_files, bval_files, bvec_files, mask_files, 
                            site_names=site_names, sub_ids=sub_ids, sub_dirs=sub_dirs)



