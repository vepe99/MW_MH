import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ytree
import yt
yt.enable_parallelism()
from tqdm import  tqdm

a = ytree.load('../data/y_tree_data/ahf_halos/snap_N64L16_000.parameter')

df = pd.DataFrame(columns = ['merge_index', 'redshift', 'merge_header', 'mass_merge_header', 'mass_progenitor_parallel_to_merge_header', 'merge_branch', 'progenitor_parallel_to_merge_branch'])

#first we get the main progenitor of the arbor, this is in root -> leaves order
# progenitor_root_to_leaves = isolated_TreeNode['prog']

merge_index = 0
#now we prune the arbor and keep the branch that are not the progenitor of the arbor, we will need the index i
# for i in tqdm(range( len(progenitor_leaves_to_root)) ):
for node in ytree.parallel_tree_nodes(a[0], group='prog'):

    #if the difference in the number of nodes bewteen two consecutive node in the progenitor_leaves_to_root is bigger than 1 it means that there is another branch:
    l_i = node.tree_size
    if l_i >3:
        l_i_old = list(node['tree'])[1].tree_size
        if l_i - l_i_old > 1:

            pruned_branches = [j for j in node['tree'] if j['uid'] not in list(node['tree'])[1]['tree', 'uid'] ]
            merge_header = [j for j in pruned_branches if j['redshift'] == list(node['tree'])[1]['redshift'] ]
            for m_h in merge_header:
                merge_branch = [j for j in pruned_branches if j['uid'] in m_h['tree', 'uid']]
                redshift_merge_branch = [j['redshift'] for j in merge_branch]

                progenitor_parallel_to_merge_branch = [j for j in list(node['tree'])[1]['tree'] if j['redshift'] <= max(redshift_merge_branch) and j['redshift'] >= min(redshift_merge_branch) ]

                df.loc[len(df)] = [merge_index, m_h['redshift'], merge_header, m_h['mass'], list(node['tree'])[1]['mass'], merge_branch[::-1], progenitor_parallel_to_merge_branch]

            merge_index += 1

df['merge_index'] = df['merge_index'].max() - df['merge_index']
# df = df.sort_values(ascending=False, by=['merge_index'])
# df.to_csv('./merger_history_parallel/')



