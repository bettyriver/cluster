#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:06:48 2025

calculate metallicity within 1Re

workflow:
    1. get BPT, select SF spaxels
    2. emission line dust correction
    3. combine emission lines flux of SF spaxels within 1 Re
    4. metallicity calculations for multiple diagnostics
    5. save the data
    
    
simplify workflow:
    
    1. get the average metallicity within 1Re --> can directly write in 
       metallicity_SAMI.py

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


def make_metal_csv(catid,z,foreground_E_B_V,savepath):
    
    # get the flux of all emission lines needed
    
    emi_list = ['Halpha','Hbeta','NII6583','OII3728','OIII5007','SII6716',
                'SII6731']
    
    # Dictionary to hold data
    emission_data = {}
    
    for line in emi_list:
        flux, flux_err = read_emi_dc(catid=catid, emi_line=line)
        emission_data[line] = flux
        emission_data[line+'_err'] = flux_err
    
    # Dictionary to data after dust correction
    emission_data_correction = {}
    
    # get the dust corrected flux for all emission lines
    for line in emi_list:
        
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
    
    
    
    radius_kpc, re_kpc = get_dist_map(catid)
    
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

def make_metal_csv_mc(catid,z,foreground_E_B_V,savepath):
    
    # get the flux of all emission lines needed
    
    emi_list = ['Halpha','Hbeta','NII6583','OII3728','OIII5007','SII6716',
                'SII6731']
    
    # Dictionary to hold data
    emission_data = {}
    
    for line in emi_list:
        flux, flux_err = read_emi_dc(catid=catid, emi_line=line)
        emission_data[line] = flux
        emission_data[line+'_err'] = flux_err
    
    # Dictionary to data after dust correction
    emission_data_correction = {}
    
    # get the dust corrected flux for all emission lines
    for line in emi_list:
        
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
    
    
    
    radius_kpc, re_kpc = get_dist_map(catid)
    
    pixel_select = (SF_region) & (radius_kpc <= re_kpc)
    
    ## to do: decide the minimum number of pixel to keep this galaxy
    # skip this galaxy if less than 3 pixels is SF
    if pixel_select.sum() < 3:
        return None
    
    
    n2s2ha, n2s2ha_err = calculate_metallicity_sum_mc(
        emission_data_dic=emission_data_correction, 
        met_diagnostic='N2S2Ha_D16', pixel_select=pixel_select)
    scal, scal_err = calculate_metallicity_sum_mc(
        emission_data_dic=emission_data_correction, 
        met_diagnostic='Scal_PG16', pixel_select=pixel_select)
    n2o2, n2o2_err = calculate_metallicity_sum_mc(
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



def calculate_metallicity_sum_mc(emission_data_dic, met_diagnostic,pixel_select,
                                 n_realiations=1000):
    '''
    calculate metallicity uncertainty using Monte Carlo error propagation

    Parameters
    ----------
    emission_data_dic : TYPE
        DESCRIPTION.
    met_diagnostic : TYPE
        DESCRIPTION.
    pixel_select : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    emission_data_dic_orig = emission_data_dic
    
    if met_diagnostic == "N2S2Ha_D16":
        # N2S2Ha - Dopita et al. (2016)
        nii = emission_data_dic['NII6583']
        sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
        ha = emission_data_dic['Halpha']
        
        nii_err = emission_data_dic['NII6583_err']
        sii_err = np.sqrt(emission_data_dic['SII6716_err']**2 +
                          emission_data_dic['SII6731_err']**2)
        ha_err = emission_data_dic['Halpha_err']
        
        good_pts = (nii/nii_err >= 1) & (sii/sii_err >=1)
        
        pixel_select = pixel_select & good_pts
        
        z_stack = []
        
        
        for i in range(n_realiations):
            emission_data_dic = randomize_emission(emission_data_dic_orig)
            nii = emission_data_dic['NII6583']
            sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
            ha = emission_data_dic['Halpha']
            
            nii = np.nansum(nii[pixel_select])
            sii = np.nansum(sii[pixel_select])
            ha = np.nansum(ha[pixel_select])
            
            
            logR = np.log10(nii / sii) + 0.264 * np.log10(nii / ha)
            logOH12 = 8.77 + logR + 0.45 * (logR + 0.3)**5
            
            z_stack.append(logOH12)
        return np.nanmedian(z_stack), np.nanstd(z_stack)
    
    
    elif met_diagnostic=="Scal_PG16":
        oiii = emission_data_dic['OIII5007'] * (1 + 1.0/3.0)
        sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
        nii = emission_data_dic['NII6583'] * (1 + 1.0/3.0)
        hb = emission_data_dic['Hbeta']
        
        oiii_err = emission_data_dic['OIII5007_err'] * (1 + 1.0/3.0)
        sii_err = np.sqrt(emission_data_dic['SII6716_err']**2 +
                          emission_data_dic['SII6731_err']**2)
        nii_err = emission_data_dic['NII6583_err'] * (1 + 1.0/3.0)
        hb_err = emission_data_dic['Hbeta_err']
        
        
        pixel_select = pixel_select & (oiii/oiii_err>=1) & (sii/sii_err>=1) \
            & (nii/nii_err>=1) & (hb/hb_err>=1)
        z_stack = []
        
        
        for i in range(n_realiations):
            emission_data_dic = randomize_emission(emission_data_dic_orig)
            oiii = emission_data_dic['OIII5007'] * (1 + 1.0/3.0)
            sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
            nii = emission_data_dic['NII6583'] * (1 + 1.0/3.0)
            hb = emission_data_dic['Hbeta']
            
            
            oiii = np.nansum(oiii[pixel_select])
            sii = np.nansum(sii[pixel_select])
            nii = np.nansum(nii[pixel_select])
            hb = np.nansum(hb[pixel_select])
            
            # Scal - Pilyugin & Grebel (2016)
            logO3S2 = np.log10((oiii) / (sii))  # Their R3/S2
            logN2Hb = np.log10((nii) / hb)  # Their N2 
            logS2Hb = np.log10((sii) / hb)  # Their S2

            # Decide which branch we're on
            logOH12 = np.full_like(logO3S2, np.nan)
            #pts_lower = logN2Hb < -0.6
            #pts_upper = logN2Hb >= -0.6
            
            if logN2Hb < -0.6:

                logOH12 = 8.072 + 0.789 * logO3S2 + 0.726 * logN2Hb + ( 1.069 - 0.170 * logO3S2 + 0.022 * logN2Hb) * logS2Hb
            else:
                logOH12 = 8.424 + 0.030 * logO3S2 + 0.751 * logN2Hb + (-0.349 + 0.182 * logO3S2 + 0.508 * logN2Hb) * logS2Hb
            z_stack.append(logOH12)
        return np.nanmedian(z_stack), np.nanstd(z_stack)
    
    if met_diagnostic=='N2O2_K19':
        # N2O2 - Kewley 2019
        nii = emission_data_dic['NII6583']
        oii = emission_data_dic['OII3728'] 
        
        
        nii_err = emission_data_dic['NII6583_err']
        oii_err = emission_data_dic['OII3728_err']
        
        good_pts = (nii/nii_err >= 1) & (oii/oii_err >=1)
        
        pixel_select = pixel_select & good_pts
        z_stack = []
        
        
        for i in range(n_realiations):
            emission_data_dic = randomize_emission(emission_data_dic_orig)
            nii = emission_data_dic['NII6583']
            oii = emission_data_dic['OII3728'] 
            
            nii = np.nansum(nii[pixel_select])
            oii = np.nansum(oii[pixel_select])
            
            
            
            lognii_oii = np.log10(nii/oii)
            
            x = lognii_oii
            y = -3.17
            
            
            # Calculate z
            z = (9.4772 + 1.1797 * x + 0.5085 * y + 0.6879 * x * y +
                 0.2807 * x**2 + 0.1612 * y**2 + 0.1187 * x * y**2 +
                 0.1200 * y * x**2 + 0.2293 * x**3 + 0.0164 * y**3)
            z_stack.append(z)
        return np.nanmedian(z_stack), np.nanstd(z_stack)
    
    
    
def randomize_emission(emission_data, seed=42, clip_negative=True):
    """
    Given a dict with keys 'line' and 'line_err',
    return a new dict with fluxes perturbed by Gaussian errors.
    
    Parameters
    ----------
    emission_data : dict
        Dictionary with keys like 'Halpha', 'Halpha_err', ...
    seed : int or None
        Random seed for reproducibility.
    clip_negative : bool
        If True, negative draws are clipped to 0.
    
    Returns
    -------
    dict
        New dictionary with perturbed fluxes only (no *_err keys).
    """
    rng = np.random.default_rng(seed)
    new_data = {}

    for key in emission_data:
        if key.endswith('_err'):
            continue  # skip error arrays
        flux = emission_data[key]
        err  = emission_data.get(key + '_err', None)
        if err is None:
            raise ValueError(f"Missing error for line {key}")

        # Draw random Gaussian noise
        perturbed = rng.normal(loc=flux, scale=err)

        if clip_negative:
            perturbed = np.clip(perturbed, 0, None)

        new_data[key] = perturbed

    return new_data


    
def calculate_metallicity_sum(emission_data_dic, met_diagnostic,pixel_select):
    if met_diagnostic == "N2S2Ha_D16":
        # N2S2Ha - Dopita et al. (2016)
        nii = emission_data_dic['NII6583']
        sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
        ha = emission_data_dic['Halpha']
        
        nii_err = emission_data_dic['NII6583_err']
        sii_err = np.sqrt(emission_data_dic['SII6716_err']**2 +
                          emission_data_dic['SII6731_err']**2)
        ha_err = emission_data_dic['Halpha_err']
        
        good_pts = (nii/nii_err >= 1) & (sii/sii_err >=1)
        
        pixel_select = pixel_select & good_pts
        
        nii = np.nansum(nii[pixel_select])
        sii = np.nansum(sii[pixel_select])
        ha = np.nansum(ha[pixel_select])
        
        nii_err = np.sqrt(np.nansum(nii_err[pixel_select]**2))
        sii_err = np.sqrt(np.nansum(sii_err[pixel_select]**2))
        ha_err = np.sqrt(np.nansum(ha_err[pixel_select]**2))
        
        logR = np.log10(nii / sii) + 0.264 * np.log10(nii / ha)
        logOH12 = 8.77 + logR + 0.45 * (logR + 0.3)**5
        #good_pts = (-1.1 < logR) & (logR < 0.5)  # Limits eyeballed from their fig. 3
        #good_pts = good_pts & SF_region
        #logOH12[~good_pts] = np.nan
        
        
        # Partial derivatives for error propagation
        ln10 = np.log(10)
        
        # Error of log(nii/sii)
        dlog_nii_sii_dnii = 1 / (ln10 * nii)
        dlog_nii_sii_dsii = -1 / (ln10 * sii)
        var_x = (dlog_nii_sii_dnii * sii)**2 * nii_err**2 + (dlog_nii_sii_dsii * nii)**2 * sii_err**2
        
        # Error of log(nii/ha)
        dlog_nii_ha_dnii = 1 / (ln10 * nii)
        dlog_nii_ha_dha = -1 / (ln10 * ha)
        var_y = (dlog_nii_ha_dnii * ha)**2 * nii_err**2 + (dlog_nii_ha_dha * nii)**2 * ha_err**2
        
        # Total error on logR
        var_logR = var_x + (0.264**2) * var_y
        logR_err = np.sqrt(var_logR)
        # Derivative of logOH12 w.r.t. logR
        dlogOH12_dlogR = 1 + 2.25 * (logR + 0.3)**4
        
        # Error propagation
        logOH12_err = dlogOH12_dlogR * logR_err
        #logOH12_err[~good_pts] = np.nan 
        return logOH12, logOH12_err 
    
    elif met_diagnostic=="Scal_PG16":
        oiii = emission_data_dic['OIII5007'] * (1 + 1.0/3.0)
        sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
        nii = emission_data_dic['NII6583'] * (1 + 1.0/3.0)
        hb = emission_data_dic['Hbeta']
        
        oiii_err = emission_data_dic['OIII5007_err'] * (1 + 1.0/3.0)
        sii_err = np.sqrt(emission_data_dic['SII6716_err']**2 +
                          emission_data_dic['SII6731_err']**2)
        nii_err = emission_data_dic['NII6583_err'] * (1 + 1.0/3.0)
        hb_err = emission_data_dic['Hbeta_err']
        
        
        pixel_select = pixel_select & (oiii/oiii_err>=1) & (sii/sii_err>=1) \
            & (nii/nii_err>=1) & (hb/hb_err>=1)
        
        oiii = np.nansum(oiii[pixel_select])
        sii = np.nansum(sii[pixel_select])
        nii = np.nansum(nii[pixel_select])
        hb = np.nansum(hb[pixel_select])
        
        oiii_err = np.sqrt(np.nansum(oiii_err[pixel_select]**2))
        sii_err = np.sqrt(np.nansum(sii_err[pixel_select]**2))
        nii_err = np.sqrt(np.nansum(nii_err[pixel_select]**2))
        hb_err = np.sqrt(np.nansum(hb_err[pixel_select]**2))
        
        
        # Scal - Pilyugin & Grebel (2016)
        logO3S2 = np.log10((oiii) / (sii))  # Their R3/S2
        logN2Hb = np.log10((nii) / hb)  # Their N2 
        logS2Hb = np.log10((sii) / hb)  # Their S2

        # Decide which branch we're on
        logOH12 = np.full_like(logO3S2, np.nan)
        #pts_lower = logN2Hb < -0.6
        #pts_upper = logN2Hb >= -0.6
        
        if logN2Hb < -0.6:

            logOH12 = 8.072 + 0.789 * logO3S2 + 0.726 * logN2Hb + ( 1.069 - 0.170 * logO3S2 + 0.022 * logN2Hb) * logS2Hb
        else:
            logOH12 = 8.424 + 0.030 * logO3S2 + 0.751 * logN2Hb + (-0.349 + 0.182 * logO3S2 + 0.508 * logN2Hb) * logS2Hb
        
        #logOH12[~SF_region] = np.nan
        
        
        #### calculate error
        # Error propagation for logs: log10(a/b) = log10(a) - log10(b)
        ln10 = np.log(10)
        
        logO3S2_err = np.sqrt((1 / (ln10 * oiii))**2 * oiii_err**2 +
                              (1 / (ln10 * sii))**2 * sii_err**2)
        
        logN2Hb_err = np.sqrt((1 / (ln10 * nii))**2 * nii_err**2 +
                              (1 / (ln10 * hb))**2 * hb_err**2)
        
        logS2Hb_err = np.sqrt((1 / (ln10 * sii))**2 * sii_err**2 +
                              (1 / (ln10 * hb))**2 * hb_err**2)
        logOH12_err = np.full_like(logO3S2, np.nan)
        
        
        if logN2Hb < -0.6:
        
            # Lower branch formula
            logR = logO3S2
            logN = logN2Hb
            logS = logS2Hb
            dlogR = logO3S2_err
            dlogN = logN2Hb_err
            dlogS = logS2Hb_err
            # Error propagation for lower branch
            dZ_dR = 0.789 - 0.170 * logS
            dZ_dN = 0.726 + 0.022 * logS
            dZ_dS = 1.069 - 0.170 * logR + 0.022 * logN
            
            logOH12_err = np.sqrt(
                (dZ_dR * dlogR)**2 +
                (dZ_dN * dlogN)**2 +
                (dZ_dS * dlogS)**2
            )
        
        else:
        
            # Upper branch formula
            logR = logO3S2
            logN = logN2Hb
            logS = logS2Hb
            dlogR = logO3S2_err
            dlogN = logN2Hb_err
            dlogS = logS2Hb_err
            
            
            # Error propagation for upper branch
            dZ_dR = 0.030 + 0.182 * logS
            dZ_dN = 0.751 + 0.508 * logS
            dZ_dS = -0.349 + 0.182 * logR + 0.508 * logN  
            
            logOH12_err = np.sqrt(
                (dZ_dR * dlogR)**2 +
                (dZ_dN * dlogN)**2 +
                (dZ_dS * dlogS)**2
            )
            
            #logOH12_err[~SF_region] = np.nan
        
        return logOH12, logOH12_err     
    if met_diagnostic=='N2O2_K19':
        # N2O2 - Kewley 2019
        nii = emission_data_dic['NII6583']
        oii = emission_data_dic['OII3728'] 
        
        
        nii_err = emission_data_dic['NII6583_err']
        oii_err = emission_data_dic['OII3728_err']
        
        good_pts = (nii/nii_err >= 1) & (oii/oii_err >=1)
        
        pixel_select = pixel_select & good_pts
        
        nii = np.nansum(nii[pixel_select])
        oii = np.nansum(oii[pixel_select])
        
        
        nii_err = np.sqrt(np.nansum(nii_err[pixel_select]**2))
        oii_err = np.sqrt(np.nansum(oii_err[pixel_select]**2))
        
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii, b=oii, 
                                                a_err=nii_err, 
                                                b_err=oii_err)
        
        
        x = lognii_oii
        x_err = lognii_oii_err
        y = -3.17
        
        
        # Calculate z
        z = (9.4772 + 1.1797 * x + 0.5085 * y + 0.6879 * x * y +
             0.2807 * x**2 + 0.1612 * y**2 + 0.1187 * x * y**2 +
             0.1200 * y * x**2 + 0.2293 * x**3 + 0.0164 * y**3)
        
        # Calculate the partial derivative of z with respect to x
        dz_dx = (1.1797 + 0.6879 * y + 2 * 0.2807 * x + 0.1187 * y**2 +
                 2 * 0.1200 * y * x + 3 * 0.2293 * x**2)
        
        # Calculate the error in z
        z_err = np.abs(dz_dx) * x_err
        return z, z_err


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


def get_dist_map(galaxy_id):
    parent_path = '/Users/ymai0110/Documents/cluster_galaxies/'
    
    
    metal_fits_path =  parent_path +  'SAMI_metallicity/'
    #sfr_path = spaxelsleuth_path + 'sfr_related/'
    
    
    cluster_csv = pd.read_csv(parent_path+
                              'cluster_classification_from_Oguzhan/'+
                              'SAMI_DR3_Cluster_Galaxies_Oguzhan_sample_metalyifan.csv')
    
    
    metal_fits = fits.open(metal_fits_path + str(galaxy_id) + '.fits')
    
    # create dist_arr in kpc for this galaxy
    query = cluster_csv['CATID']==galaxy_id
    
    map_shape = metal_fits['SCAL'].data.shape
    
    xcen = map_shape[1]/2-0.5 # cube center and galaxy center, in the unit of pixel
    ycen = map_shape[0]/2-0.5
    
    z = cluster_csv['z_spec'][query].to_numpy()[0]
    re = cluster_csv['r_e'][query].to_numpy()[0] # r-band major axis effective radius in arcsec
    #b_a = cluster_csv['B_on_A'][query].to_numpy()[0]
    pa = cluster_csv['PA'][query].to_numpy()[0] # r-band position angle in deg
    #fwhm = cluster_csv['fwhm'][query].to_numpy()[0] # FWHM of PSF in cube in arcsec
    ellip = cluster_csv['ellip'][query].to_numpy()[0] # r-band ellipticity
    
    
    
    dist_arr = ellip_distarr(size=map_shape, 
                             centre=(xcen,ycen),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
    
    re_kpc = arcsec_to_kpc(rad_in_arcsec=re, z=z)
    return radius_kpc, re_kpc