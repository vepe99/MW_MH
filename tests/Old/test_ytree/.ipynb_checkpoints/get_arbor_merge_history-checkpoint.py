import numpy as np
import pandas as pd
import pynbody as pb
import ytree
from tqdm import tqdm


def get_arbor_merge_history(isolated_TreeNode):
    """
    
    Arbor merger history extractor
    input: it takes as input isolated Treenode like a[0] 
    output: Dataframe(columns = ['merge_index', 'redshift', 'merge_header', 'mass_merge_header', 'mass_progenitor_parallel_to_merge_header', 'merge_branch', 'progenitor_parallel_to_merge_branch'])
    
    """
    
    df = pd.DataFrame(columns = ['merge_index', 'redshift', 'merge_header', 'mass_merge_header', 'mass_progenitor_parallel_to_merge_header', 'merge_branch', 'progenitor_parallel_to_merge_branch'])

    #first we get the main progenitor of the arbor, this is in root -> leaves order
    progenitor_root_to_leaves = list(isolated_TreeNode['prog'])

    #we want the leaves -> root order for all of our data
    progenitor_leaves_to_root = progenitor_root_to_leaves[::-1]
    
   
    merge_index = 0
    #now we prune the arbor and keep the branch that are not the progenitor of the arbor, we will need the index i
    for i in tqdm(range( len(progenitor_leaves_to_root)) ):
        
        #if the difference in the number of nodes bewteen two consecutive node in the progenitor_leaves_to_root is bigger than 1 it means that there is another branch:
        l_i = progenitor_leaves_to_root[i].tree_size
        l_i_old = progenitor_leaves_to_root[i-1].tree_size
        if l_i - l_i_old > 1:
            
            pruned_branches = [j for j in progenitor_leaves_to_root[i]['tree'] if j['uid'] not in progenitor_leaves_to_root[i-1]['tree', 'uid'] ]
            merge_header = [j for j in pruned_branches if j['redshift'] == progenitor_leaves_to_root[i-1]['redshift'] ] #you want to take the branch whose progenitor is the main branch, not some others branches that are parallel
            for m_h in merge_header:
                merge_branch = [j for j in pruned_branches if j['uid'] in m_h['tree', 'uid']]
                redshift_merge_branch = [j['redshift'] for j in merge_branch]
                  
                progenitor_parallel_to_merge_branch = [j for j in progenitor_leaves_to_root[:i] if j['redshift'] <= max(redshift_merge_branch) and j['redshift'] >= min(redshift_merge_branch) ]  #this put the merge_header in the last position of the list
                
                df.loc[len(df)] = [merge_index, m_h['redshift'], merge_header, m_h['mass'], progenitor_leaves_to_root[i-1]['mass'], merge_branch[::-1], progenitor_parallel_to_merge_branch]
                
            merge_index += 1
    
    return df