from utils import *
from diff_utils import *

######################################################## Freesurfer lobe
# to remove "parahippocampal" frontalpole

Frontal_label_list_old = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
            'parsopercularis', 'parstriangularis', 'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 
            'precentral', 'paracentral', 'frontalpole', 'rostralanteriorcingulate','caudalanteriorcingulate']
Temporal_label_list_old = ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'fusiform', 
                        'transversetemporal','entorhinal', 'temporalpole', 'parahippocampal']

Frontal_label_list = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
            'parsopercularis', 'parstriangularis', 'parsorbitalis', 
            'precentral', 'paracentral', 'frontalpole', 'rostralanteriorcingulate','caudalanteriorcingulate']
Temporal_label_list = ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'fusiform', 
                        'transversetemporal','entorhinal', 'parahippocampal']

Parietal_label_list = ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus']
Occipital_label_list = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine', 'posteriorcingulate',  'isthmuscingulate']
Cingulate_label_list = ['rostralanteriorcingulate','caudalanteriorcingulate', 'posteriorcingulate',  'isthmuscingulate']

lobe_mapping_dict_old = {'frontal': Frontal_label_list_old, 'parietal': Parietal_label_list, 'temporal': Temporal_label_list_old, 'occipital': Occipital_label_list}
lobe_mapping_dict = {'all': Frontal_label_list + Parietal_label_list + Temporal_label_list + Occipital_label_list + Cingulate_label_list, 
                    'frontal': Frontal_label_list, 'parietal': Parietal_label_list, 'temporal': Temporal_label_list, 'occipital': Occipital_label_list}
######################################################## Freesurfer lobe


def gen_roi_dict(freesurf_file):
    '''Generate the label-roiname dictionary for the freesurf_file'''

    roi_label_list = []
    roi_name_list = []
    df = pd.read_fwf(freesurf_file, skiprows=2, delimiter="\t", lineterminator='\n')
    indx_and_roinames = [df.loc[i][0] for i in range(len(df)) if '#' not in df.loc[i][0]]

    for indx_and_roiname in indx_and_roinames:
        indx_and_roiname_split = [i for i in indx_and_roiname.split(' ') if i != '']
        roi_label_list.append(int(indx_and_roiname_split[0]))
        roi_name_list.append(indx_and_roiname_split[1])
        
    return dict(zip(roi_label_list, roi_name_list))


def gen_label_dict(freesurf_file):

    free_roi_dicts = gen_roi_dict(freesurf_file)
    free_label_dicts = {v: k for k, v in free_roi_dicts.items()}

    return free_label_dicts


def extract_roi(ana_file, labels, save_roi_file=None):
    
    ana_img = load(ana_file).get_fdata().astype(int)
    
    # extract roi
    if type(labels) is list:
        roi_img = np.any(np.array([ana_img == label for label in labels]), axis=0)
    else:
        roi_img = (ana_img == labels)
    
    if save_roi_file is not None:
        if not exists(save_roi_file):
            save_nifti(save_roi_file, roi_img, ref_fname=ana_file)
    else:
        return roi_img
    
# def merge_roi(roi_files, save_roi_file=None):
    
#     rois_stack = np.array([load(roi_file).get_fdata() > 0 for roi_file in roi_files])
#     print(rois_stack.shape)
#     roi_img = np.sum(rois_stack, axis=0)
    
#     if save_roi_file is not None:
#         save_nifti(save_roi_file, roi_img, ref_fname=roi_files[0])
#     else:
#         return roi_img