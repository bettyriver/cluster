#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 13:58:30 2025

cluster metallicity analysis from spaxelsleuth data

@author: ymai0110
"""
import numpy as np
import astropy.io.fits as fits
import pandas as pd
from math import ceil
import hector_spaxelsleuth as hss


def get_bin_metallicity(metal_map,error_map,BPT_map,dist_arr,bin_size):
    '''
    get the bin metallicity of equal weighted and inverse variance weighted
    versions. 

    Parameters
    ----------
    metal_map : 2d-array
        metallicity map of any diagnostic.
    error_map : 2d-array
        map of metallicity error
    BPT_map : 2d-array
        BPT map from spaxelsleuth. use for selecting SF region. 0.0 is SF
    dist_arr : 2d-array
        map shows the distance to the centre. in kpc
    bin_size : float
        The size of metallicity bin. in kpc

    Returns
    -------
    metal_mean_arr: 2-d array
        equal weighted mean metallicity of each bin
    metal_mean_err_arr: 2-d array
        error of equal weighted metallicity
    metal_inv_ave_arr: 2-d array
        inverse variance weighted average metallicity of each bin
    metal_inv_ave_err_arr: 2-d array
        

    '''
    
    # mask non-SF region, only select data from SF
    
    SF_query = BPT_map==0.0
    
    metal_map_mask = metal_map[SF_query]
    error_map_mask = error_map[SF_query]
    dist_arr_mask = dist_arr[SF_query]
    
    dist_max  = np.nanmax(dist_arr_mask)
    
    #    no SF pixel    , no metal measurement in SF pixel for this diagnostic
    if np.isnan(dist_max) or np.isnan(np.nansum(metal_map_mask)):
        bin_num = ceil(dist_max/bin_size)
        
        # an array with length of bin_num and full with nan
        nan_arr = np.full(bin_num,np.nan)
        
        return nan_arr, nan_arr, nan_arr, nan_arr
    
    bin_num = ceil(dist_max/bin_size)
    
    metal_mean_arr = []
    metal_mean_err_arr = []
    metal_inv_ave_arr = []
    metal_inv_ave_err_arr = []
    
    for i in range(bin_num):
        # distance and not nan query
        dist_query = (dist_arr_mask>=i*bin_size) & (dist_arr_mask<(i+1)*bin_size)&(~np.isnan(metal_map_mask))
        metal_map_mask_query = metal_map_mask[dist_query]
        error_map_mask_query = error_map_mask[dist_query]
        #  no SF pixel in this bin,       no metal measurement in this bin
        if len(metal_map_mask_query)<1 or np.isnan(np.nansum(metal_map_mask_query)):
            metal_mean_arr.append(np.nan)
            metal_mean_err_arr.append(np.nan)
            metal_inv_ave_arr.append(np.nan)
            metal_inv_ave_err_arr.append(np.nan)
            continue
        
        
        # average 1, equal weighted
        
        metal_mean_eq = np.nanmean(metal_map_mask_query)
        metal_mean_err_eq = np.sqrt(np.nansum(error_map_mask_query**2)) / np.count_nonzero(~np.isnan(error_map_mask_query))
        
        metal_mean_arr.append(metal_mean_eq)
        metal_mean_err_arr.append(metal_mean_err_eq)
        
        # average 2, inverse variance weighted
        
        weights = 1 / error_map_mask_query**2
        ave_inv_var = np.nansum(metal_map_mask_query * weights) / np.count_nonzero(~np.isnan(weights))
        ave_inv_var_err = np.sqrt(1 / np.nansum(weights))
        
        metal_inv_ave_arr.append(ave_inv_var)
        metal_inv_ave_err_arr.append(ave_inv_var_err)
        
        
        
    metal_mean_arr = np.array(metal_mean_arr)
    metal_mean_err_arr = np.array(metal_mean_err_arr)
    metal_inv_ave_arr = np.array(metal_inv_ave_arr)
    metal_inv_ave_err_arr = np.array(metal_inv_ave_err_arr)        

    
    
    
    return metal_mean_arr,metal_mean_err_arr,metal_inv_ave_arr,metal_inv_ave_err_arr




def make_bin_metal_table(metal_fits,sfr_fits,dist_arr,bin_size):
    '''
    Make a table that includes binned metallicity for all diagnostics
    
    Each column is a dignostic and each row is 0.25 Re/0.5 PSF bin. The number of 
    row depends on the BPT map of galaxy.
    
    Parameters
    ----------
    metal_fits: astropy.fits
        FITS document for metallicity from spaxelsleuth. It includes 
        metallicity measured using different diagnostics.
    sfr_fits: astropy.fits
        FITS document for SFR related from spaxelsleuth. It includes BPT map.
    dist_arr: numpy.2d-array
        2d-array that gives the distance to the centre
    bin_size: float
        The size of bin in kpc

    Returns
    -------
    metal_table: pandas.DataFrame
        a dataframe that each row is one bin, each column is a diagnostic

    '''
    
    all_diag = hss.Spaxelsleuth.metal_ext_names
    
    BPT_map = sfr_fits['BPT (numeric) (total)'].data
    
    
    # check if any SF spaxel, if not, simply return 0
    has_SF = np.any(BPT_map == 0)
    
    if has_SF is False:
        print('no SF spaxel in this galaxy')
        return 0
    
    # create bin dist arr for the whole table
    SF_query = BPT_map==0.0
    dist_arr_mask = dist_arr[SF_query]
    dist_max  = np.nanmax(dist_arr_mask)
    bin_num = ceil(dist_max/bin_size)
    bin_arr_kpc = []
    for i in range(bin_num):
        bin_arr_kpc.append(i*bin_size+0.5*bin_size)
    bin_arr_kpc = np.array(bin_arr_kpc)
    
    
    df = pd.DataFrame({'bin_arr_kpc':bin_arr_kpc})
    
    # get bin metallicity for all diagnostics, loop 
    
    
    
    for metal_diag in all_diag:
        err_16 = metal_diag[:-7] + 'error (lower) ' + metal_diag[-7:]
        err_84 = metal_diag[:-7] + 'error (upper) ' + metal_diag[-7:]
        
        err_16_map = metal_fits[err_16].data
        err_84_map = metal_fits[err_84].data
    
        err_map = (err_16_map + err_84_map)/2
        
        metal_map = metal_fits[metal_diag].data
        
        # ext name for this diagnostic
        diag_name = metal_diag[15:-9]
        ave1_name = diag_name + '(eq)'
        ave1_err_name = diag_name + '(eq_err)'
        
        ave2_name = diag_name + '(inv)'
        ave2_err_name = diag_name + '(inv_err)'

        a1, e1, a2, e2 = get_bin_metallicity(
                            metal_map=metal_map,
                            error_map=err_map,
                            BPT_map=BPT_map,
                            dist_arr=dist_arr,
                            bin_size=bin_size)
        
        df[ave1_name] = a1
        df[ave1_err_name] = e1
        df[ave2_name] = a2
        df[ave2_err_name] = e2
    
    
    return df

def get_table_bulk(galaxy_list,bin_type):
    '''
    for galaxies in the list, get bin metallicity for all diagnostics

    Parameters
    ----------
    galaxy_list : list of str
        list of galaxy id (str). note this name should include both ID and 
        identifier (e.g. C9001A)
    bin_type: str
        'Re' or 'PSF'. If 'Re', the bin_size is 0.25 Re. If 'PSF', the bin
        size is 0.5 PSF.

    Returns
    -------
    None.

    '''
    parent_path = '/Users/ymai0110/Documents/cluster_galaxies/'
    
    spaxelsleuth_path = parent_path + 'spaxelsleuth/'
    metal_fits_path =  spaxelsleuth_path + 'metallicity/'
    sfr_path = spaxelsleuth_path + 'sfr_related/'
    
    
    
    for galaxy_id in galaxy_list:
        metal_fits = fits.open(metal_fits_path + galaxy_id + '_metalicity.fits')
        sfr_fits = fits.open(sfr_path + galaxy_id + '_sfr_related.fits')
        # create dist_arr in kpc for this galaxy
        
        
        
        
        
        # calculate the bin size in kpc depends on the bin type