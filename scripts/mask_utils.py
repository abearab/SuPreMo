import numpy as np
import sys
sys.path.insert(0, './scripts/')
from bin_utils import get_bin

def mask_matrices(REF_pred, ALT_pred, SVTYPE, SVLEN, var_rel_pos, bin_size, offset, target_length_cropped):

    '''
    This applied to non-BND predicted matrices.
    
    Mask reference and alternate predicted matrices based on the type of variant.
    
    '''


    if SVTYPE in ['DEL', 'DUP']:
        
        # Insertions: Add nans to reference matrix and crop ends, then mirror nans to alternate matrix
        
        if SVTYPE == 'DEL':
            
            # Deletions: Same but swapping reference and alternate values
            REF_pred, ALT_pred = ALT_pred, REF_pred
            var_rel_pos.reverse()
            
            
        # Get variant relative positions in the unmasked predicted maps
        var_start_REF = get_bin(var_rel_pos[0], bin_size, offset)

        var_start_ALT = get_bin(var_rel_pos[1], bin_size, offset)
        var_end_ALT = get_bin(var_rel_pos[1] + SVLEN, bin_size, offset)

        

        #### Mask REF map
        
        REF_pred_masked = REF_pred.copy()
        ALT_pred_masked = ALT_pred.copy()

        
        # Insert the empty bins in the reference matrix where the variant is in the alternate matrix
        to_add = var_end_ALT - var_start_ALT

        for j in range(var_start_REF, var_start_REF + to_add): # range only includes the first variable 
            REF_pred_masked = np.insert(REF_pred_masked, j, np.nan, axis = 0)
            # REF_pred_masked = np.insert(REF_pred_masked, j, np.nan, axis = 1)

            
        # Crop the outside of the reference matrix 
        to_remove_left = var_start_REF - var_start_ALT
        to_remove_right = len(REF_pred_masked) - target_length_cropped - to_remove_left

        if to_remove_left != 0:
            REF_pred_masked = REF_pred_masked[to_remove_left:]
        if to_remove_right != 0:
            REF_pred_masked = REF_pred_masked[:-to_remove_right]



        
        if SVTYPE == 'DEL':
            
            # Deletions: Swap reference and alternate values back
            REF_pred_masked, ALT_pred_masked = ALT_pred_masked, REF_pred_masked
        
        
        
    
    # Inversions: Mask REF and mirror nans to ALT
    
    elif SVTYPE == 'INV':
        
        var_start = get_bin(var_rel_pos[0], bin_size, offset)
        var_end = get_bin(var_rel_pos[0] + SVLEN, bin_size, offset)
 

        # Mask REF map: make variant bin(s) nan 
        
        REF_pred_masked = REF_pred.copy()

        REF_pred_masked[var_start:var_end + 1] = np.nan
        # REF_pred_masked[:, var_start:var_end + 1] = np.nan

        
        # Mask ALT map: make all nans in REF_pred also nan in ALT_pred
        
        REF_pred_novalues = REF_pred_masked.copy()

        REF_pred_novalues[np.invert(np.isnan(REF_pred_novalues))] = 0

        ALT_pred_masked = ALT_pred + REF_pred_novalues

        
        
    if len(REF_pred_masked) != target_length_cropped:
        print(len(REF_pred_masked))
        print(target_length_cropped)
        raise ValueError('Masked reference matrix is not the right size.')    
        
        
    if len(ALT_pred_masked) != target_length_cropped:
        raise ValueError('Masked alternate matrix is not the right size.')
    
    
    return REF_pred_masked, ALT_pred_masked
