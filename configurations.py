import os

# TODO: change the following path based on your dataset
# should be one of the "ISBI", "MICCAI16_1", "MICCAI16_2", "MICCAI16_3", "UMCL"
PATH_DATASET = '/gpfs23/scratch/liuh26/ModDropPlusPlus-main/ms_data'
DATASETS = ["UMCL"]  # training domains

# We assume the format of file names in this pattern {PREFIX}_{PATIENT-ID}_{TIMEPOINT-ID}_{MODALITY(MASK)}.{SUFFIX}
# PREFIX: can be any string
# PATIENT-ID: number id of the patient
# TIMEPOINT-ID: number id of the timepoint
# MODALITY, MASK: t1, flair, t2, pd, mask1, mask2, etc
# SUFFIX: nii, nii.gz, etc
# e.g. training_01_01_flair.nii, training_03_05_mask1.nii
# TODO: change the following constants based on your dataset

MODALITIES = ['t1', 'flair', 't2', 'pd', 'ce']  # general
MASKS = ['mask1', 'mask2', 'mask']
SUFFIX = 'nii.gz'

# The axis corresponding to axial, sagittal and coronal, respectively
# TODO: change the following axes based on your dataset
AXIS_TO_TAKE = [2, 0, 1]


# training indepedent modes (IM)

#---- UMCL ----
# MODALITIES = ['t1', 'flair', 't2', 'ce']  # 1234
# MODALITIES = ['flair', 't2', 'ce']  # 234
# MODALITIES = ['t1', 't2', 'ce']  # 134  
# MODALITIES = ['t1', 'flair', 'ce']  # 124
# MODALITIES = ['t1', 'flair', 't2']  # 123
# MODALITIES = ['t2', 'ce']  # 34
# MODALITIES = ['flair', 'ce']  # 24
# MODALITIES = ['flair', 't2']  # 23  
# MODALITIES = ['t1', 'ce']  # 14
# MODALITIES = ['t1', 't2']  # 13
# MODALITIES = ['t1']  # 1  
# MODALITIES = ['flair']  # 2
# MODALITIES = ['t2']  # 3
# MODALITIES = ['ce']  # 4

#---- ISBI ----
# MODALITIES = ['t1', 'flair', 't2', 'pd']  # 1234
# MODALITIES = ['flair', 't2', 'pd']  # 234
# MODALITIES = ['t1', 't2', 'pd']  # 134  
# MODALITIES = ['t1', 'flair', 'pd']  # 124
# MODALITIES = ['t1', 'flair', 't2']  # 123
# MODALITIES = ['t2', 'pd']  # 34
# MODALITIES = ['flair', 'pd']  # 24
# MODALITIES = ['flair', 't2']  # 23  
# MODALITIES = ['t1', 'pd']  # 14
# MODALITIES = ['t1', 't2']  # 13
# MODALITIES = ['t1', 'flair']  # 12
# MODALITIES = ['t1']  # 1  
# MODALITIES = ['flair']  # 2 
# MODALITIES = ['t2']  # 3
# MODALITIES = ['pd']  # 4