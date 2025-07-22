import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import math

random.seed(2025)

workdir = '../RGPRec-Ability/dataA_result/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tensorflow', help='dataset name: Tensorflow/SourceForge')
parser.add_argument('--sigma', default='0', help='social strength definition')
parser.add_argument('--test_prop', default=0.1, help='the proportion of dataA_result used for test')
args = parser.parse_args()

# load dataA_result
if args.dataset == 'Tensorflow':
	click_f = loadmat(workdir + 'Tensorflow/rating.mat')['rating']
	trust_f = loadmat(workdir + 'Tensorflow/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'SourceForge':
	click_f = loadmat(workdir + 'SourceForge/rating.mat')['rating']
	trust_f = loadmat(workdir + 'SourceForge/trustnetwork.mat')['trustnetwork']
else:
	pass 

click_list = []
trust_list = []

d_avg_list = []
d_items_list = []
d_items_divlist = []

d_user_simlist = []
d_user_items_list = []
d_user_items_divlist = []

p_avg_list = []
p_user_list = []
p_user_divlist = []

usereloper_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	did = s[0]
	pid = s[1]
	if args.dataset == 'Tensorflow':
		label = s[3]
	elif args.dataset == 'SourceForge':
		label = s[3]


	if did > usereloper_count:
		usereloper_count = did
	if pid > item_count:
		item_count = pid
	if label > rate_count:
		rate_count = label
	click_list.append([did, pid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

pos_list = list(set(pos_list))  

random.shuffle(pos_list) 
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open(workdir + args.dataset + '/dataset_'+ args.sigma +'.pkl', 'wb') as f: 
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns=['did', 'pid', 'label'])
valid_df = pd.DataFrame(valid_set, columns=['did', 'pid', 'label'])
test_df = pd.DataFrame(test_set, columns=['did', 'pid', 'label'])

click_df = pd.DataFrame(click_list, columns=['did', 'pid', 'label'])
train_df = train_df.sort_values(axis=0, ascending=True, by='did')
all_avg = train_df['label'].mean()


for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['pid'] == i]
	p_ratings = hist['label'].tolist()
	if p_ratings == []:
		p_avg_list.append(all_avg)
	else:
		p_avg_list.append(hist['label'].mean())

for u in tqdm(range(usereloper_count + 1)):
	hist = train_df[train_df['did'] == u]
	d_ratings = hist['label'].tolist()
	if d_ratings == []:
		d_avg_list.append(all_avg)
	else:
		d_avg_list.append(hist['label'].mean())

for u in tqdm(range(usereloper_count + 1)):
	hist = train_df[train_df['did'] == u]
	d_items = hist['pid'].tolist()
	d_ratings = hist['label'].tolist()
	if d_items == []:
		d_items_divlist.append([(0, 0)])
	else:
		d_items_divlist.append([(pid, round(abs(rating-p_avg_list[pid]))) for pid, rating in zip(d_items, d_ratings)])

for u in tqdm(range(usereloper_count + 1)):
	hist = train_df[train_df['did'] == u]
	d_items = hist['pid'].tolist()
	d_ratings = hist['label'].tolist()
	if d_items == []:
		d_items_list.append([(0, 0)])
	else:
		d_items_list.append([(pid, rating) for pid, rating in zip(d_items, d_ratings)])

train_df = train_df.sort_values(axis=0, ascending=True, by='pid')


for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['pid'] == i]
	p_user = hist['did'].tolist()
	p_ratings = hist['label'].tolist()
	if p_user == []:
		p_user_divlist.append([(0, 0)])
	else:
		p_user_divlist.append([(did, round(abs(rating-d_avg_list[did]))) for did, rating in zip(p_user, p_ratings)])

for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['pid'] == i]
	p_user = hist['did'].tolist()
	p_ratings = hist['label'].tolist()
	if p_user == []:
		p_user_list.append([(0, 0)])
	else:
		p_user_list.append([(did, rating) for did, rating in zip(p_user, p_ratings)])

for s in trust_f:
	did = s[0]
	fid = s[1]
	if did > usereloper_count or fid > usereloper_count:
		continue

	trust_list.append([did, fid])

trust_df = pd.DataFrame(trust_list, columns=['did', 'fid'])
trust_df = trust_df.sort_values(axis=0, ascending=True, by='did')


d_user_similar = []
for u in tqdm(range(usereloper_count + 1)):
	d_d_similar = []
	d_info = dict(d_items_list[u])
	hist = trust_df[trust_df['did'] == u]
	d_user = hist['fid'].unique().tolist()
	if d_user == []:
		d_user_similar.append([(0, 0)])
		d_user_items_list.append([[(0, 0)]])
		d_user_items_divlist.append([[0, 0]])
	else:
		for usereloper in d_user:
			usereloper_info = dict(d_items_list[usereloper])
			inter_list = list(set(usereloper_info.keys()).intersection(set(d_info.keys())))
			inter_count = len(inter_list)
			for item in inter_list:
				if abs(d_info[item]-usereloper_info[item]) > int(args.sigma):
					inter_count = inter_count - 1
			d_d_similar.append((usereloper, inter_count+1))
		if d_d_similar == []:
			d_user_similar.append([(0, 0)])
			d_user_items_list.append([[(0, 0)]])
			d_user_items_divlist.append([[(0, 0)]])
		else:
			d_user_similar.append(d_d_similar)
			dd_items = []
			dd_items_div = []
			for (did, inter_count) in d_d_similar:
				dd_items.append(d_items_list[did])
				dd_items_div.append(d_items_divlist[did])
			d_user_items_list.append(dd_items)
			d_user_items_divlist.append(dd_items_div)


with open(workdir + args.dataset +'/list_'+ args.sigma +'.pkl', 'wb') as f:
	pickle.dump(d_items_divlist, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(d_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(d_avg_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(d_user_similar, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(d_user_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(d_user_items_divlist, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(p_avg_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(p_user_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(p_user_divlist, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((usereloper_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)
