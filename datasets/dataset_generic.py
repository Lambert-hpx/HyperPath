from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False, out_test=None):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if out_test is not None:
		out_test = pd.Series(out_test, name='out_test')
		splits.append(out_test)
		column_keys.append('out_test')
	# import ipdb;ipdb.set_trace()

	# test_df = pd.read_csv('/home/yyhuang/WSI/Dataset/Camelyon16/test_label.csv')
	# splits[0] = splits[0]._append(splits[1]).reset_index(drop=True)
	# splits[1] = splits[2]
	# splits[2] = test_df['slide_id']
	# import ipdb;ipdb.set_trace()
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = column_keys)
	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		survival_pred = False,
		output_slide_ids = False,
		sites = None,
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		self.output_slide_ids = output_slide_ids
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		# import ipdb;ipdb.set_trace()
		if sites is not None:
			slide_data['site'] = slide_data['case_id'].apply(lambda x: x[5:7])
			slide_data = slide_data[slide_data['site'].isin(sites)]
			# site_df = pd.read_csv('/home/yyhuang/WSI/Dataset/Camelyon16/camelyon16_site.csv')
			# slide_data = pd.merge(slide_data, site_df, on='slide_id', how='inner')
			# slide_data = slide_data[slide_data['site'].isin(sites)]
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, label_num=0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'label_num': label_num,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}
		# import ipdb;ipdb.set_trace()
		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)
		# import ipdb;ipdb.set_trace()
		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)
		# import ipdb;ipdb.set_trace()

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):

		# import ipdb;ipdb.set_trace()
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			return train_split, val_split, test_split
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			try:
				out_test_split = self.get_split_from_df(all_splits, 'out_test')
			except:
				out_test_split = test_split
			# out_test_split = test_split
			# import ipdb;ipdb.set_trace()
			
			return [train_split, val_split, test_split, out_test_split]

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		# self.use_h5 = False
		self.use_h5 = True

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		# if not self.use_h5:
		# 	if self.data_dir:
		# 		full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
		# 		features = torch.load(full_path)
		# 		# print(features.shape)
		# 		return features, label
			
		# 	else:
		# 		return slide_id, label

		# else:
		full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
		with h5py.File(full_path,'r') as hdf5_file:
			# print(hdf5_file)
			# import ipdb;ipdb.set_trace()
			if hdf5_file.get('features2') is None:
				features2 = hdf5_file['features'][:]
			else:
				features2 = hdf5_file['features2'][:]
			features = hdf5_file['features'][:]
			
			coords = hdf5_file['coords'][:]

		features = torch.from_numpy(features).float()
		features2 = torch.from_numpy(features2).float()
		# if self.output_slide_ids:
		return features, features2, label, slide_id, coords
		# return features, features2, label

class Generic_HMIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir,  
		**kwargs):
	
		super(Generic_HMIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		# self.use_h5 = False
		self.use_h5 = True

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def return_splits(self, from_id=True, csv_path=None):

		# import ipdb;ipdb.set_trace()
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split_H(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split_H(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split_H(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			return train_split, val_split, test_split
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			try:
				out_test_split = self.get_split_from_df(all_splits, 'out_test')
			except:
				out_test_split = test_split
			# out_test_split = test_split
			# import ipdb;ipdb.set_trace()
			
			return [train_split, val_split, test_split, out_test_split]

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)
		# import ipdb;ipdb.set_trace()

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split_H(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split
	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir
		# print(data_dir,'!!!!!')
		# import ipdb;ipdb.set_trace()
		full_path = os.path.join(data_dir,'conch_512_con','h5_files','{}.h5'.format(slide_id))
		with h5py.File(full_path,'r') as hdf5_file:
			features = hdf5_file['features'][:]
			coords = hdf5_file['coords'][:]
   
		full_path2 = os.path.join(data_dir,'conch_512_4096_con','h5_files','{}.h5'.format(slide_id))
		with h5py.File(full_path2,'r') as hdf5_file:
			features2 = hdf5_file['features'][:]
			coords2 = hdf5_file['coords'][:]
  

		features = torch.from_numpy(features).float()
		features2 = torch.from_numpy(features2).float()
		# if self.output_slide_ids:
		coords_dict = {
			'regions': coords2,
            'patches': coords
		}
		# import ipdb;ipdb.set_trace()
		# region_min_xy = np.min(coords_dict['regions'],axis=0)
		# region_max_xy = np.max(coords_dict['regions'],axis=0)
        
        # patch2region = 4096 * ((coords_dict['patches'] - region_min_xy) // 4096) + region_min_xy
		patch2region_indices = patch2region(coords_dict['patches'],coords_dict['regions'])
		patch2region_indices = torch.from_numpy(patch2region_indices)
		# import ipdb;ipdb.set_trace()
		# print(len(coords_dict['regions']),len(coords_dict['patches']))
		return features, features2, label, slide_id, patch2region_indices
		# return features, features2, label

def patch2region(patch_coords,region_coords):
    # 将 coords 和 coords2 扩展为可以广播的形状
	# coords: (N, 1, 2), coords2: (1, M, 2)
	patch_coords_expanded = patch_coords[:, np.newaxis, :]  # Shape: (N, 1, 2)
	region_coords_expanded = region_coords[np.newaxis, :, :]  # Shape: (1, M, 2)

	# 计算每个 patch 是否在 region 的范围内
	# 条件: region_x <= patch_x <= region_x + 4096 且 region_y <= patch_y <= region_y + 4096
	x_condition = (patch_coords_expanded[:, :, 0] >= region_coords_expanded[:, :, 0]) & \
				(patch_coords_expanded[:, :, 0] + 512 <= region_coords_expanded[:, :, 0] + 4096)
	y_condition = (patch_coords_expanded[:, :, 1] >= region_coords_expanded[:, :, 1]) & \
				(patch_coords_expanded[:, :, 1] + 512 <= region_coords_expanded[:, :, 1] + 4096)

	# 合并 x 和 y 的条件
	within_region = x_condition & y_condition  # Shape: (N, M)

	# 找到每个 patch 对应的 region 索引
	# 如果某个 patch 属于多个 region，只取第一个满足条件的 region
	patch_region_indices = np.argmax(within_region, axis=1)  # Shape: (N,)

	# 将没有满足条件的 patch 标记为 -1
	patch_region_indices[np.sum(within_region, axis=1) == 0] = -1
	return patch_region_indices
class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)

class Generic_Split_H(Generic_HMIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


