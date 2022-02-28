#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def hierarchy_pos_original(G, root=None, width=.5, vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    From: https://epidemicsonnetworks.readthedocs.io/en/latest/_modules/EoN/auxiliary.html#hierarchy_pos
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos


def merge_func(transition_matrix, n_cluster, motif_norm, merge_sel):
    
    if merge_sel == 0:
        # merge nodes with highest transition probability
        cost = np.max(transition_matrix)
        merge_nodes = np.where(cost == transition_matrix)
        
    if merge_sel == 1:
            
        cost_temp = 100
        for i in range(n_cluster):
            for j in range(n_cluster):
                try:
                    cost = (motif_norm[i] + motif_norm[j]) / np.abs(transition_matrix[i,j] + transition_matrix[j,i] )
                except IndexError:
                    print("Error: Transition probabilities between motif "+str(i)+" and motif "+str(j)+ " are zero.")
                if cost <= cost_temp:
                    cost_temp = cost
                    merge_nodes = (np.array([i]), np.array([j]))                
    return merge_nodes


def graph_to_tree(motif_usage, transition_matrix, n_cluster, merge_sel=1):
    
    if merge_sel == 1:
        # motif_usage_temp = np.load(path_to_file+'/behavior_quantification/motif_usage.npy')
        motif_usage_temp = motif_usage
        motif_usage_temp_colsum = motif_usage_temp.sum(axis=0)
        motif_norm = motif_usage_temp/motif_usage_temp_colsum
        motif_norm_temp = motif_norm.copy()
    else:
        motif_norm_temp = None
    
    merging_nodes = []
    hierarchy_nodes = []
    trans_mat_temp = transition_matrix.copy()
    is_leaf = np.ones((n_cluster), dtype='int')
    node_label = []
    leaf_idx = []
    
    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1)==0)
        reduction = len(temp) + 1
    else:
        reduction = 1
    
    for i in range(n_cluster-reduction):
    
#        max_tr = np.max(trans_mat_temp) #merge function
#        nodes = np.where(max_tr == trans_mat_temp)
        nodes = merge_func(trans_mat_temp, n_cluster, motif_norm_temp, merge_sel)
        
        if np.size(nodes) >= 2:
            nodes = np.array([nodes[0][0], nodes[1][0]])
        
        if is_leaf[nodes[0]] == 1:
            is_leaf[nodes[0]] = 0
            node_label.append('leaf_left_'+str(i))
            leaf_idx.append(1)
        
        elif is_leaf[nodes[0]] == 0:
            node_label.append('h_'+str(i)+'_'+str(nodes[0]))
            leaf_idx.append(0)
            
        if is_leaf[nodes[1]] == 1:
            is_leaf[nodes[1]] = 0
            node_label.append('leaf_right_'+str(i))
            hierarchy_nodes.append('h_'+str(i)+'_'+str(nodes[1]))            
            leaf_idx.append(1)
            
        elif is_leaf[nodes[1]] == 0:
            node_label.append('h_'+str(i)+'_'+str(nodes[1]))
            hierarchy_nodes.append('h_'+str(i)+'_'+str(nodes[1]))
            leaf_idx.append(0)
            
        merging_nodes.append(nodes)

        node1_trans_x = trans_mat_temp[nodes[0],:]
        node2_trans_x = trans_mat_temp[nodes[1],:]
        
        node1_trans_y = trans_mat_temp[:,nodes[0]]
        node2_trans_y = trans_mat_temp[:,nodes[1]]
        
        new_node_trans_x = node1_trans_x + node2_trans_x
        new_node_trans_y = node1_trans_y + node2_trans_y
        
        trans_mat_temp[nodes[1],:] = new_node_trans_x
        trans_mat_temp[:,nodes[1]] = new_node_trans_y
        
        trans_mat_temp[nodes[0],:] = 0
        trans_mat_temp[:,nodes[0]] = 0
        
        trans_mat_temp[nodes[1],nodes[1]] = 0
        
        if merge_sel == 1:
            motif_norm_1 = motif_norm_temp[nodes[0]]
            motif_norm_2 = motif_norm_temp[nodes[1]]
            
            new_motif = motif_norm_1 + motif_norm_2
            
            motif_norm_temp[nodes[0]] = 0
            motif_norm_temp[nodes[1]] = 0
            
            motif_norm_temp[nodes[1]] = new_motif

    merge = np.array(merging_nodes)
#    merge = np.concatenate((merge),axis=1).T
    
    T = nx.Graph()
      
    T.add_node('Root')
    node_dict = {}
    
    if leaf_idx[-1] == 0:
        temp_node = 'h_'+str(merge[-1,1])+'_'+str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1,1]] = temp_node
        
    if leaf_idx[-1] == 1:
        T.add_edge(merge[-1,1], 'Root')
        
    if leaf_idx[-2] == 0:
        temp_node = 'h_'+str(merge[-1,0])+'_'+str(28)
        T.add_edge(temp_node, 'Root')
        node_dict[merge[-1,0]] = temp_node
        
    if leaf_idx[-2] == 1:
        T.add_edge(merge[-1,0], 'Root')
        
    idx = len(leaf_idx)-3
    
    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1)==0)
        reduction = len(temp) + 2
    else:
        reduction = 2
        
    for i in range(n_cluster-reduction)[::-1]:
        
        if leaf_idx[idx-1] == 1:
            if merge[i,1] in node_dict:
                T.add_edge(merge[i,0], node_dict[merge[i,1]])
            else:
                T.add_edge(merge[i,0], temp_node)
            
        if leaf_idx[idx] == 1:
            if merge[i,1] in node_dict:
                T.add_edge(merge[i,1], node_dict[merge[i,1]])
            else:
                T.add_edge(merge[i,1], temp_node)
            
        if leaf_idx[idx] == 0:
            new_node = 'h_'+str(merge[i,1])+'_'+str(i)
            if merge[i,1] in node_dict:
                T.add_edge(node_dict[merge[i,1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
#            node_dict[merge[i,1]] = new_node
            
            if leaf_idx[idx-1] == 1:
                temp_node = new_node
                node_dict[merge[i,1]] = new_node
            else:
                new_node_2 = 'h_'+str(merge[i,0])+'_'+str(i)
#                temp_node = 'h_'+str(merge[i,0])+'_'+str(i)
                T.add_edge(node_dict[merge[i,1]], new_node_2)
#                node_dict[merge[i,0]] = temp_node
                node_dict[merge[i,1]] = new_node
                node_dict[merge[i,0]] = new_node_2
#                temp_node = new_node
                
        elif leaf_idx[idx-1] == 0:
            new_node = 'h_'+str(merge[i,0])+'_'+str(i)
            if merge[i,1] in node_dict:
                T.add_edge(node_dict[merge[i,1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
            node_dict[merge[i,0]] = new_node
            
            if leaf_idx[idx] == 1:
                temp_node = new_node
            else:
                new_node = 'h_'+str(merge[i,1])+'_'+str(i)
                T.add_edge(temp_node, new_node)
                node_dict[merge[i,1]] = new_node
                temp_node = new_node
                
        idx -= 2
        
    return T


def draw_tree(T, file):
    # pos = nx.drawing.layout.fruchterman_reingold_layout(T)
    pos = hierarchy_pos(T,'Root',width=.5, vert_gap = 0.1, vert_loc = 0)#, xcenter = 50 
    fig = plt.figure()
    nx.draw_networkx(T, pos)  
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if not os.path.exists('trees/'):
       os.mkdir('trees')
    fig.savefig('trees/'+file+'_tree.png') 
    plt.close('all')


def _traverse_tree(T, node, traverse_preorder,traverse_list):
    traverse_preorder += str(node[0])
    traverse_list.append(node[0])
    children = list(T.neighbors(node[0]))
    
    if len(children) == 3:
#        print(children)
        for child in children:
            if child in traverse_list:
#                    print(child)
                children.remove(child)
        
    if len(children) > 1:
        traverse_preorder += '{'
        traverse_preorder_temp = _traverse_tree(T, [children[0]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
         
        traverse_preorder += '}{'
        
        traverse_preorder_temp = _traverse_tree(T, [children[1]], '',traverse_list)
        traverse_preorder += traverse_preorder_temp
        traverse_preorder += '}'
    
    return traverse_preorder
    
def traverse_tree(T, root_node=None):
    if not root_node:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    traverse_preorder = '{'
    traverse_preorder = _traverse_tree(T, node, traverse_preorder,traverse_list)
    traverse_preorder += '}'
    
    return traverse_preorder


def _traverse_tree_cutline(T, node, traverse_list, cutline, level, community_bag, community_list=None): 
    cmap = plt.get_cmap("tab10")
    traverse_list.append(node[0])
    if community_list and not isinstance(node[0], str):
        community_list.append(node[0])
    children = list(T.neighbors(node[0]))
    
    if len(children) == 3:
        children = [x for x in children if x not in traverse_list]
                    
    if len(children) > 1:
        if nx.shortest_path_length(T,'Root',node[0])==cutline:
            #create new list
            traverse_list1 = []
            traverse_list2 = []
            community_bag = _traverse_tree_cutline(T, [children[0]], traverse_list, cutline, level+1, community_bag, traverse_list1)
            community_bag = _traverse_tree_cutline(T, [children[1]], traverse_list, cutline, level+1, community_bag, traverse_list2)
            joined_list=traverse_list1+traverse_list2
            community_bag.append(joined_list)
            if not isinstance(node[0], str): #append itself
                community_bag.append([node[0]])            
        else:
             community_bag = _traverse_tree_cutline(T, [children[0]], traverse_list, cutline, level+1, community_bag, community_list)
             community_bag = _traverse_tree_cutline(T, [children[1]], traverse_list, cutline, level+1, community_bag, community_list)            

    return  community_bag


def traverse_tree_cutline(T, root_node=None, cutline=2, fill=False, n_cluster=15):
    if root_node == None:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    color_map = []
    community_bag=[]
    level = 0
    community_bag = _traverse_tree_cutline(T, node, traverse_list,cutline, level, color_map,community_bag)
    if fill:
        import itertools
        used = list(itertools.chain.from_iterable(community_bag))
        missing = [x for x in range(n_cluster) if x not in used]
        for i in missing:
            community_bag.extend([[i]])
    return community_bag
    
