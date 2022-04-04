import os.path
import torchvision.transforms as transforms
import numpy as np
import json
import nibabel as nib
from data.base_dataset import BaseDataset
from configurations import *
from util.image_property import hash_file, normalize_image, slice_with_neighborhood


def get_domain_code(path):
    code = np.zeros(5)
    if 'ISBI' in path:
        code[0] = 1
    elif 'MICCAI16_1' in path:
        code[1] = 1
    elif 'MICCAI16_2' in path:
        code[2] = 1
    elif 'MICCAI16_3' in path:
        code[3] = 1
    elif 'UMCL' in path:
        code[4] = 1
    return code


def get_modality_code(modalities):
    code = np.zeros(5)
    if 't1' in modalities:
        code[0] = 1
    if 'flair' in modalities:
        code[1] = 1
    if 't2' in modalities:
        code[2] = 1
    if 'pd' in modalities:
        code[3] = 1
    if 'ce' in modalities:
        code[4] = 1
    return code


def get_3d_paths(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(MODALITIES[0]+'.' + SUFFIX):
                images.append([])
                images[-1].append(os.path.join(root, fname))
                for i in range(1, len(MODALITIES)):
                    images[-1].append(os.path.join(root, fname.replace(MODALITIES[0]+'.'+SUFFIX, MODALITIES[i]+'.' + SUFFIX)))
                images[-1].append(os.path.join(root, fname.replace(MODALITIES[0] + '.' + SUFFIX, 'mask.' + SUFFIX)))
    images.sort(key=lambda x: x[0])
    return images


def flip_by_times(np_array, times):
    for i in range(times):
        np_array = np.flip(np_array, axis=1)
    return np_array


class Ms3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.all_paths = []
        self.dic_properties = {}
        for dataset_name in DATASETS:
            self.dir_AB = os.path.join(opt.dataroot, dataset_name, opt.phase)
            self.all_paths += get_3d_paths(self.dir_AB)
            with open(os.path.join(opt.dataroot, dataset_name, 'properties.json'), 'r') as f:
                self.dic_properties.update(json.load(f))
        self.neighbors = opt.input_nc // 2

    def __getitem__(self, index):
        paths_this_scan = self.all_paths[index]
        # print(paths_this_scan): [../_t1.nii.gz, ../_flair.nii.gz, ../_t2.nii.gz, ../_pd.nii.gz, ../_mask.nii.gz]
        voxel_sizes = nib.affines.voxel_sizes(nib.load(paths_this_scan[0]).affine)
        data_all_modalities = {}
        all_modalities = []
        if os.path.exists(paths_this_scan[-1]):
            data_all_modalities['mask'] = nib.load(paths_this_scan[-1]).get_fdata()
        path_name = paths_this_scan[0][:-9]

        for i, modality in enumerate(MODALITIES):
            # modality = 't1', 'flair', 't2', 'pd', 'ce'
            # i        =   0,     1,      2,    3,    4
            path_modality = path_name + modality + '.nii.gz'
            if os.path.exists(path_modality): # and '_t2' not in path_modality and '_t1' not in path_modality and '_flair' not in path_modality:
                all_modalities.append(modality)
                label_modality = hash_file(path_modality)
                data_modality = nib.load(path_modality).get_fdata()
                
                if label_modality in self.dic_properties:
                    peak_modality = self.dic_properties[label_modality]['peak']
                else:
                    peak_modality = normalize_image(data_modality, modality)
                data_all_modalities[modality] = np.array(data_modality / peak_modality, dtype=np.float32)
            else:
                data_all_modalities[modality] = np.zeros(data_all_modalities['mask'].shape)

        data_return = {mod: {'axial': [], 'sagittal': [], 'coronal': []} for mod in MODALITIES+['mask']}
        data_return['org_size'] = {'axial': None, 'sagittal': None, 'coronal': None}
        data_return['mask_paths'] = paths_this_scan[-1]
        data_return['alt_paths'] = paths_this_scan[0]
        data_return['dc'] = get_domain_code(paths_this_scan[0])
        data_return['mc'] = get_modality_code(all_modalities)

        for k, orientation in enumerate(['axial', 'sagittal', 'coronal']):
            ratio = [size for axis, size in enumerate(voxel_sizes) if axis != AXIS_TO_TAKE[k]]
            ratio = ratio[1] / ratio[0]
            cur_shape = data_all_modalities[MODALITIES[0]].shape
            slices_per_image = cur_shape[AXIS_TO_TAKE[k]]
            data_return['org_size'][orientation] = \
                tuple([axis_len for axis, axis_len in enumerate(cur_shape) if axis != AXIS_TO_TAKE[k]])
            for i in range(slices_per_image):
                for modality in MODALITIES:
                    slice_modality = slice_with_neighborhood(data_all_modalities[modality], AXIS_TO_TAKE[k], i, self.neighbors, ratio)
                    slice_modality = transforms.ToTensor()(slice_modality).float()
                    if modality in all_modalities:
                        slice_modality = slice_modality / 2 - 1
                    data_return[modality][orientation].append(slice_modality)
                if os.path.exists(paths_this_scan[-1]):
                    slice_modality = slice_with_neighborhood(data_all_modalities['mask'], AXIS_TO_TAKE[k], i, 0)
                    slice_modality = transforms.ToTensor()(slice_modality).float()
                    slice_modality = slice_modality * 2 - 1
                    data_return['mask'][orientation].append(slice_modality)
        return data_return

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'Ms3dDataset'
