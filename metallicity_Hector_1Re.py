#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:47:03 2025

@author: ymai0110
"""

import sys
sys.path.insert(0,'/Users/ymai0110/Documents/myPackages/metalpy/')
from metal_uncertainty import intrinsic_flux_with_err
from BPT import bptregion
from rps_cluster import read_emi_dc
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import ceil
from post_spaxelsleuth import get_bin_metallicity, calculate_gradient
from metal_uncertainty import log10_ratio_with_error
from metalgradient import ellip_distarr, pix_to_kpc, arcsec_to_kpc
from metallicity_SAMI_1Re import calculate_metallicity_sum, calculate_metallicity_sum_mc

class EmissionLines:
    wavelengths = {
        'Halpha': 6562.8,
        'Hbeta': 4861.3,
        'NII6583': 6583.4,
        'OII3728': 3728.0,
        'OIII5007': 5006.8,
        'SII6716': 6716.4,
        'SII6731': 6730.8
    }

    @classmethod
    def get_wavelength(cls, line_name):
        if line_name not in cls.wavelengths:
            raise ValueError(f"Unknown emission line: {line_name}")
        return cls.wavelengths[line_name]
    
def read_emi_hector(catid,emi_line,datapath_other=None):
    '''
    read the emission line product for Hector data

    Parameters
    ----------
    catid : int
        ID for SAMI galaxies.
    emi_line: str
        name of emission line, e.g. Halpha, Hbeta, 
        NII6583, OIII5007

    Returns
    -------
    emi_map: 2-d array
        the 2-d flux map of given emission line
    emi_err_map: 2-d array
        the 2-d flux error map of given emission line

    '''
    
    datapath = '/Users/ymai0110/Documents/cluster_galaxies/Hector_data/'
    
    if datapath_other is not None:
        datapath = datapath_other
    
    datapath += catid + '_reccomp.fits.gz'
    if not os.path.exists(datapath):
        return None, None
    fits_file = fits.open(datapath)
    emi_map = fits_file[emi_line].data
    emi_err_map = fits_file[emi_line+'_ERR'].data
    
    if emi_map.ndim==3:
        return emi_map[0], emi_err_map[0]
    else:
        return emi_map, emi_err_map

def make_metal_csv(catid,z,foreground_E_B_V,savepath,csv_path=None,
                   emi_datapath=None):
    '''
    

    Parameters
    ----------
    catid : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    foreground_E_B_V : TYPE
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.
    csv_path : str, optional
        csv that have Re, PA, ellip data. The default is None.
    

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    # get the flux of all emission lines needed
    
    emi_list = ['HALPHA','HBETA','NII6583','OII3729','OIII5007','SII6716',
                'SII6731']
    
    emi_list_sami = ['Halpha','Hbeta','NII6583','OII3728','OIII5007','SII6716',
                'SII6731']
    
    # Dictionary to hold data
    emission_data = {}
    
    for line,line_ in zip(emi_list,emi_list_sami):
        flux, flux_err = read_emi_hector(catid=catid, emi_line=line,datapath_other=emi_datapath)
        if flux is None:
            print(str(catid)+' file not exists.')
            return None
        emission_data[line_] = flux
        emission_data[line_+'_err'] = flux_err
    
    # Dictionary to data after dust correction
    emission_data_correction = {}
    
    # get the dust corrected flux for all emission lines
    for line in emi_list_sami:
        
        # flux that need correction
        flux = emission_data[line]
        flux_err = emission_data[line+'_err']
        
        ha = emission_data['Halpha']
        ha_err = emission_data['Halpha_err']
        hb = emission_data['Hbeta']
        hb_err = emission_data['Hbeta_err']
        
        
        flux_corr, flux_err_corr = intrinsic_flux_with_err(
                                        flux_obs=flux,
                                        flux_obs_err=flux_err,
                                        wave=EmissionLines.get_wavelength(line),
                                        ha=ha,
                                        hb=hb,
                                        ha_err=ha_err,
                                        hb_err=hb_err,
                                        foreground_E_B_V=foreground_E_B_V,
                                        z=z,
                                        dust_model='F19')
        
        emission_data_correction[line] = flux_corr
        emission_data_correction[line+'_err'] = flux_err_corr
    
    
    
    
    
    #### BPT calculation ####
    
    # no need to use corrected emissioin lines for BPT, as those pairs are
    # closed to each other
    AGN, CP, SF_region = get_SF_region(emission_data)
    
    
    
    radius_kpc, re_kpc = get_dist_map(catid,csv_path=csv_path)
    
    pixel_select = (SF_region) & (radius_kpc <= re_kpc)
    
    ## to do: decide the minimum number of pixel to keep this galaxy
    # skip this galaxy if less than 3 pixels is SF
    if pixel_select.sum() < 3:
        return None
    
    
    n2s2ha, n2s2ha_err = calculate_metallicity_sum(
        emission_data_dic=emission_data_correction, 
        met_diagnostic='N2S2Ha_D16', pixel_select=pixel_select)
    scal, scal_err = calculate_metallicity_sum(
        emission_data_dic=emission_data_correction, 
        met_diagnostic='Scal_PG16', pixel_select=pixel_select)
    n2o2, n2o2_err = calculate_metallicity_sum(
        emission_data_dic=emission_data_correction, 
        met_diagnostic='N2O2_K19', pixel_select=pixel_select)
    
    df = pd.DataFrame({'CATID':catid,
                       'N2S2HA':n2s2ha,
                       'N2S2HA_err':n2s2ha_err,
                       'SCAL':scal,
                       'SCAL_err':scal_err,
                       'N2O2':n2o2,
                       'N2O2_err':n2o2_err}, index=[0])
    df.to_csv(savepath+str(catid)+'.csv')
    
    return df

def get_SF_region(emission_data_dic):
    ha_err = emission_data_dic['Halpha_err']
    hb_err = emission_data_dic['Hbeta_err']
    oiii_err = emission_data_dic['OIII5007_err']
    nii_err = emission_data_dic['NII6583_err']
    
    ha = emission_data_dic['Halpha']
    hb = emission_data_dic['Hbeta']
    oiii = emission_data_dic['OIII5007']
    nii = emission_data_dic['NII6583']
    
    ha_snr = ha/ha_err
    hb_snr = hb/hb_err
    oiii_snr = oiii/oiii_err
    nii_snr = nii/nii_err
    
    crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
    crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
    indplot =  crit_err & crit_snr
    
    
    
    x = np.log10(nii/ha)
    y = np.log10(oiii/hb)
    
    ##constrction construction coordinates###
    nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
    ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    
    x_type = np.full_like(xpos, np.nan)
    y_type = np.full_like(xpos, np.nan)
    x_type[indplot] = x[indplot]
    y_type[indplot] = y[indplot]
    AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
    
    return AGN, CP, SF

def get_dist_map(galaxy_id,csv_path=None):
    
    
    datapath = '/Users/ymai0110/Documents/cluster_galaxies/Hector_data/'
    
    datapath += galaxy_id + '_reccomp.fits.gz'
    
    fits_file = fits.open(datapath)
    map_shape = fits_file['HALPHA'].data.shape[1:]

    if csv_path is None:
        parent_path = '/Users/ymai0110/Documents/cluster_galaxies/'
        cluster_csv = pd.read_csv(parent_path+
                                  'cluster_classification_from_Oguzhan/'+
                                  'Hector_DR_v0_02_cluster_members_best_cubes.csv')
    else:
        cluster_csv = pd.read_csv(csv_path)
    
    
    
    # create dist_arr in kpc for this galaxy
    query = cluster_csv['catid']==galaxy_id
    
    
    
    xcen = map_shape[1]/2-0.5 # cube center and galaxy center, in the unit of pixel
    ycen = map_shape[0]/2-0.5
    
    z = cluster_csv['z'][query].to_numpy()[0]
    re = cluster_csv['Re'][query].to_numpy()[0] # r-band major axis effective radius in arcsec
    #b_a = cluster_csv['B_on_A'][query].to_numpy()[0]
    pa = cluster_csv['PA'][query].to_numpy()[0] # r-band position angle in deg
    #fwhm = cluster_csv['fwhm'][query].to_numpy()[0] # FWHM of PSF in cube in arcsec
    ba = cluster_csv['B_on_A'][query].to_numpy()[0]
    #ellip = cluster_csv['ellip'][query].to_numpy()[0] # r-band ellipticity
    
    ellip = 1 - ba
    
    dist_arr = ellip_distarr(size=map_shape, 
                             centre=(xcen,ycen),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
    
    re_kpc = arcsec_to_kpc(rad_in_arcsec=re, z=z)
    return radius_kpc, re_kpc
