import os
import json
import shutil
import pickle
import numpy as np
import nibabel as nib
from util.util import mkdir
from util.image_property import hash_file, slice_with_neighborhood
from configurations import *


def get_domain_code(dataroot):
    code = np.zeros(5)
    if dataroot.endswith('ISBI'):
        code[0] = 1
    elif dataroot.endswith('MICCAI16_1'):
        code[1] = 1
    elif dataroot.endswith('MICCAI16_2'):
        code[2] = 1
    elif dataroot.endswith('MICCAI16_3'):
        code[3] = 1
    elif dataroot.endswith('UMCL'):
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


def remove_folder_if_exist(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print('%s removed' % folder_path)
    else:
        print('%s does not exist' % folder_path)


class DataGenerator:
    def __init__(self):
        self.sample_2d_count = 0
        self.sample_3d_count = 0
        self.dataroot = PATH_DATASET
        self.dataset_names = DATASETS
        self.dir_this_phase = None
        self.axes = [0, 1, 2]
        self.rotation_2d = -1

    def build_dataset(self, val_index, test_index, test_phases):
        val, test = 'val' in test_phases, 'test' in test_phases

        for dataset_name in self.dataset_names:
            print(f"Start generating data for dataset: {dataset_name}...")
            self.domain_code = get_domain_code(dataset_name)
            self.sample_2d_count, self.sample_3d_count = 0, 0

            with open(os.path.join(self.dataroot, dataset_name, 'properties.json'), 'r') as f:
                self.dic_properties = json.load(f)
            with open(os.path.join(self.dataroot, dataset_name, 'ids.json'), 'r') as f:
                self.dic_ids = json.load(f)

            remove_folder_if_exist(os.path.join(self.dataroot, dataset_name, 'train'))
            remove_folder_if_exist(os.path.join(self.dataroot, dataset_name, 'val'))
            remove_folder_if_exist(os.path.join(self.dataroot, dataset_name, 'test'))

            all_ids = sorted(list(map(str, self.dic_ids.keys())))
            total_num = len(all_ids)
            num_fold = 5
            splits = [total_num // num_fold] * num_fold
            for i in range(total_num % num_fold):
                splits[i] += 1
            val_ids = [all_ids[i] for i in range(sum(splits[:val_index]), sum(splits[:val_index+1]))] if val else []
            test_ids = [all_ids[i] for i in range(sum(splits[:test_index]), sum(splits[:test_index+1]))] if test else []
            train_ids = [i for i in all_ids if i not in val_ids + test_ids]

            print(val_ids, test_ids)
            self.generate_general_data('val', val_ids, '3d', dataset_name)
            self.generate_general_data('test', test_ids, '3d', dataset_name)
            self.generate_general_data('train', train_ids, '2d', dataset_name)

    def generate_general_data(self, phase, ids, mode, dataset_name, neighborhood=1):
        self.dir_this_phase = os.path.join(self.dataroot, dataset_name,  phase)  # /UMCL/train
        mkdir(self.dir_this_phase)
        for subject_id in ids:
            timepoints = self.dic_ids[str(subject_id)].keys()
            for timepoint in timepoints:
                masks = self.dic_ids[str(subject_id)][str(timepoint)]['mask'].keys()
                if mode == '3d':
                    self.sample_3d_count += 1
                for mask in masks:
                    if mode == '3d':
                        self.generate_3d_data(subject_id, timepoint, mask, phase)
                    else:
                        self.generate_2d_data(subject_id, timepoint, mask, neighborhood)

        if mode == '2d':
            seq = np.arange(0, self.sample_2d_count)
            np.random.shuffle(seq)
            # print(seq)
            for i, org in enumerate(seq):
                file_org = os.path.join(self.dir_this_phase, 'tmp_%d.pkl' % org)
                file_new = os.path.join(self.dir_this_phase, '%d.pkl' % i)
                shutil.move(file_org, file_new)

    def generate_2d_data(self, subject_id, timepoint, mask, neighborhood):
        modalities = self.dic_ids[str(subject_id)][str(timepoint)]['modalities']  # available modalities here
        path_mask = self.dic_ids[str(subject_id)][str(timepoint)]['mask'][mask]
        image_mask = nib.load(path_mask)
        voxel_sizes = nib.affines.voxel_sizes(image_mask.affine)

        # append mask into image_data
        image_data = {'mask': np.array(image_mask.get_fdata(), dtype=np.float32)}

        for modality in modalities:
            path_modality = self.dic_ids[str(subject_id)][str(timepoint)]['modalities'][modality]
            hash_label = hash_file(path_modality)
            modality_peak = self.dic_properties[hash_label]['peak']
            image_data[modality] = np.array(nib.load(path_modality).get_fdata() / modality_peak, dtype=np.float32)

        modality_code = get_modality_code(modalities)
        data_to_save = {i: [] for i in modalities}
        data_to_save['mask'] = []
        for axis in self.axes:
            ratio = [k for i, k in enumerate(voxel_sizes) if i != axis]
            ratio = ratio[0] / ratio[1]  # if there is no rot90 following, it should be ratio[1]/ratio[0]
            slices_per_image = image_data['mask'].shape[axis]
            print("Slices per image %d, current samples %d" % (slices_per_image, self.sample_2d_count))
            for i in range(slices_per_image):
                slice_mask = slice_with_neighborhood(image_data['mask'], axis, i, 0)
                if np.count_nonzero(slice_mask) < 2:
                    continue
                data_to_save['mask'] = np.rot90(slice_mask, self.rotation_2d)  # legacy due to saving png
                for modality in modalities:
                    slice_modality = slice_with_neighborhood(image_data[modality], axis, i, neighborhood)
                    data_to_save[modality] = np.rot90(slice_modality, self.rotation_2d)

                data_to_save['ratio'] = ratio
                data_to_save['dc'] = self.domain_code
                data_to_save['mc'] = modality_code

                with open(os.path.join(self.dir_this_phase, 'tmp_%d.pkl' % self.sample_2d_count), 'wb') as f:
                    pickle.dump(data_to_save, f)

                self.sample_2d_count += 1

    def generate_3d_data(self, subject_id, timepoint, mask, phase):
        modalities = self.dic_ids[str(subject_id)][str(timepoint)]['modalities']
        for modality in modalities:
            path_src = self.dic_ids[str(subject_id)][str(timepoint)]['modalities'][modality]
            path_dst = os.path.join(self.dir_this_phase, mask + '_%s%d_%s.%s' % (phase, self.sample_3d_count, modality, SUFFIX))
            shutil.copyfile(path_src, path_dst)
        path_src_mask = self.dic_ids[str(subject_id)][str(timepoint)]['mask'][mask]
        path_dst_mask = os.path.join(self.dir_this_phase, mask + '_%s%d_mask.%s' % (phase, self.sample_3d_count, SUFFIX))
        shutil.copyfile(path_src_mask, path_dst_mask)


if __name__ == '__main__':
    seed = 10
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    dataroot = '/media/liuhan/HanLiu/ms_data/MICCAI16_3'