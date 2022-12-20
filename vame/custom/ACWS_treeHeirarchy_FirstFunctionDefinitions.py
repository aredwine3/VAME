#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:17:50 2022

@author: smith
"""


def traverse_tree(T, root_node=None):
    if not root_node:
        node=['Root']
    else:
        node=[root_node]
    traverse_list = []
    traverse_preorder = '{'
    
    def _traverse_tree(T, node, traverse_preorder):
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
            traverse_preorder_temp = _traverse_tree(T, [children[0]], '')
            traverse_preorder += traverse_preorder_temp
             
            traverse_preorder += '}{'
            
            traverse_preorder_temp = _traverse_tree(T, [children[1]], '')
            traverse_preorder += traverse_preorder_temp
            traverse_preorder += '}'
        
        return traverse_preorder
    
    traverse_preorder = _traverse_tree(T, node, traverse_preorder)
    traverse_preorder += '}'
    
    return traverse_preorder

