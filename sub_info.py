from utils import *

def load_abcd_info():
    '''load df of demo info for abcd dataset'''
    base_dir='/ifs/loni/faculty/shi/spectrum/yxia/dataset'
    info_df = pd.read_excel(pjoin(base_dir, 'abcd_harmonization_subject_list.xlsx'))
    basic_info_df = info_df.loc[:, ('src_subject_id','nihtbx_demo_age','sex')]
    basic_info_df.rename(columns={"src_subject_id": "subject", "nihtbx_demo_age": "age", 'sex':'gender'}, inplace=True)
    return basic_info_df


def load_hcp_info():
    '''load df of demo info for hcp dataset'''

    base_dir='/ifs/loni/faculty/shi/spectrum/yxia/dataset'
    hcp_info2_df = pd.read_csv(pjoin(base_dir, 'HCP_info2.csv'))
    hcp_info3_df = pd.read_excel(pjoin(base_dir, 'HCP_TwinInfo.xlsx'))
#     hcp_info_df = pd.merge(hcp_info3_df, hcp_info2_df,  on=['Subject','Subject'])
    hcp_info_df = hcp_info3_df.merge(hcp_info2_df, left_on='Subject', right_on='Subject')
    
    basic_hcp_info_df = hcp_info_df.loc[:, ('Subject','Age_in_Yrs','Gender')]
    basic_hcp_info_df.rename(columns={"Subject": "subject", "Age_in_Yrs": "age", 'Gender':'gender'}, inplace=True)
    basic_hcp_info_df['subject'] = basic_hcp_info_df['subject'].astype('str')
    return basic_hcp_info_df

def load_adni_info():
    '''load df of demo info for adni dataset'''

    
    gender_dict = {1:'M', 0:'F'}
    # load data info
    adni_info_df1 = pd.read_csv('/ifs/loni/faculty/shi/spectrum/yxia/dataset_harm2/ADNI_harm/code/adni_baseline_info_dti.csv')
    adni_info_df2 = pd.read_csv('/ifs/loni/faculty/shi/spectrum/yxia/dataset/ADNI_2019/2019ADNI_tau_589.csv')
    adni_info_df = adni_info_df1.merge(adni_info_df2, left_on='Subjname', right_on='Subject ID')

    basic_adni_info_df = adni_info_df.loc[:, ('Subjname','PatientAge','Sex')]
    basic_adni_info_df.rename(columns={"Subjname": "subject", "PatientAge": "age", 'Sex':'gender'}, inplace=True)
    basic_adni_info_df['subject'] = basic_adni_info_df['subject'].astype('str')
    basic_adni_info_df["age"] = basic_adni_info_df["age"].fillna('0').apply(lambda x: float(x.replace('Y', '')) if 'Y' in x else 0.0)
    basic_adni_info_df["gender"] = basic_adni_info_df["gender"].apply(lambda x: gender_dict[x])

    return basic_adni_info_df

def load_hcpd_info():
    '''load df of demo info for hcpd dataset'''

    base_dir = '/ifs/loni/faculty/shi/spectrum/yxia/dataset'
    hcpd_info_df = pd.read_csv(pjoin(base_dir, 'HCPD_info.csv'))
    basic_hcp_info_df = hcpd_info_df.loc[:, ('src_subject_id','interview_age','sex')]
    basic_hcp_info_df.rename(columns={"src_subject_id": "subject", "interview_age": "age", 'sex':'gender'}, inplace=True)
    basic_hcp_info_df = basic_hcp_info_df.drop(basic_hcp_info_df.index[0]) 
    basic_hcp_info_df["age"] = basic_hcp_info_df["age"].apply(lambda x: float(x)/12)

    return basic_hcp_info_df

def extract_age(sub_name, tar_df):
    age = tar_df.loc[tar_df['subject'] == sub_name]['age'].values[0]
    return age

def extract_gender(sub_name, tar_df):
    gender = tar_df.loc[tar_df['subject'] == sub_name]['gender'].values[0]
    return gender

def extract_sub_name(sub_name, tar_df):
    re_sub_name = tar_df.loc[tar_df['subject'] == sub_name]['subject'].values[0]
    return re_sub_name



def demo_extractor(sub_name, site_name):
    if site_name == 'HCPDevelopment':
        tar_df = load_hcpd_info()
        age = extract_age(sub_name.replace('_V1_MR', ''), tar_df)
        gender = extract_gender(sub_name.replace('_V1_MR', ''), tar_df)
    elif site_name == 'HCP' or site_name == 'HCP_7T':
        tar_df = load_hcp_info()
        age = extract_age(sub_name, tar_df)
        gender = extract_gender(sub_name, tar_df)
    elif site_name == 'GE' or site_name == 'SIEMENS':
        tar_df = load_abcd_info()
        age = extract_age(sub_name, tar_df)
        gender = extract_gender(sub_name, tar_df)
    elif site_name == 'Dismr750_54' or site_name == 'Prisma_fit_46':
        tar_df = load_adni_info()
        age = extract_age(sub_name, tar_df)
        gender = extract_gender(sub_name, tar_df)
        
    return age, gender