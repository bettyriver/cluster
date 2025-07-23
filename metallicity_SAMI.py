#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:37:29 2025

calculate metallicity map using emission line products from SAMI DR3

1. dust correction -- already done in metal_uncertianty.intrinsic_flux_with_err
2. BPT map -- done in BPT.bptregion
3. metallicity measurement using Scal and N2S2Ha diagnostics

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

def make_metal_fits(catid,z,foreground_E_B_V,savepath):
    
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
    
    AGN, CP, SF_region = get_SF_region(emission_data_correction)
    
    
    
    #### get the metallicity #####
    
    n2s2ha_map, n2s2ha_map_err = calculate_metallicity(
                            emission_data_dic=emission_data_correction, 
                            met_diagnostic='N2S2Ha_D16', 
                            SF_region=SF_region)
    
    scal_map, scal_map_err = calculate_metallicity(
                            emission_data_dic=emission_data_correction, 
                            met_diagnostic='Scal_PG16', 
                            SF_region=SF_region)
    
    
    ### save the metal map as fits ####
    
    # Create Primary HDU (can be empty or include metadata)
    primary_hdu = fits.PrimaryHDU()
    
    # Create Image HDUs for each map
    n2s2ha_hdu = fits.ImageHDU(data=n2s2ha_map, name='N2S2HA')
    n2s2ha_err_hdu = fits.ImageHDU(data=n2s2ha_map_err, name='N2S2HA_ERR')
    scal_hdu = fits.ImageHDU(data=scal_map, name='SCAL')
    scal_err_hdu = fits.ImageHDU(data=scal_map_err, name='SCAL_ERR')
    
    # Combine into HDUList
    hdul = fits.HDUList([primary_hdu, n2s2ha_hdu, n2s2ha_err_hdu, scal_hdu, scal_err_hdu])
    
    # Save to file
    hdul.writeto(savepath, overwrite=True)
    
    
    
    

def calculate_metallicity(emission_data_dic, met_diagnostic,SF_region):
    if met_diagnostic == "N2S2Ha_D16":
        # N2S2Ha - Dopita et al. (2016)
        nii = emission_data_dic['NII6583']
        sii = emission_data_dic['SII6716'] + emission_data_dic['SII6731']
        ha = emission_data_dic['Halpha']
        
        nii_err = emission_data_dic['NII6583_err']
        sii_err = np.sqrt(emission_data_dic['SII6716_err']**2 +
                          emission_data_dic['SII6731_err']**2)
        ha_err = emission_data_dic['Halpha_err']
        
        logR = np.log10(nii / sii) + 0.264 * np.log10(nii / ha)
        logOH12 = 8.77 + logR + 0.45 * (logR + 0.3)**5
        good_pts = (-1.1 < logR) & (logR < 0.5)  # Limits eyeballed from their fig. 3
        good_pts = good_pts & SF_region
        logOH12[~good_pts] = np.nan
        
        
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
        logOH12_err[~good_pts] = np.nan 
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
        
        # Scal - Pilyugin & Grebel (2016)
        logO3S2 = np.log10((oiii) / (sii))  # Their R3/S2
        logN2Hb = np.log10((nii) / hb)  # Their N2 
        logS2Hb = np.log10((sii) / hb)  # Their S2

        # Decide which branch we're on
        logOH12 = np.full_like(logO3S2, np.nan)
        pts_lower = logN2Hb < -0.6
        pts_upper = logN2Hb >= -0.6

        logOH12[pts_lower] = 8.072 + 0.789 * logO3S2[pts_lower] + 0.726 * logN2Hb[pts_lower] + ( 1.069 - 0.170 * logO3S2[pts_lower] + 0.022 * logN2Hb[pts_lower]) * logS2Hb[pts_lower]
        logOH12[pts_upper] = 8.424 + 0.030 * logO3S2[pts_upper] + 0.751 * logN2Hb[pts_upper] + (-0.349 + 0.182 * logO3S2[pts_upper] + 0.508 * logN2Hb[pts_upper]) * logS2Hb[pts_upper]
        
        logOH12[~SF_region] = np.nan
        
        
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
        
        # Lower branch formula
        logR = logO3S2[pts_lower]
        logN = logN2Hb[pts_lower]
        logS = logS2Hb[pts_lower]
        dlogR = logO3S2_err[pts_lower]
        dlogN = logN2Hb_err[pts_lower]
        dlogS = logS2Hb_err[pts_lower]
        # Error propagation for lower branch
        dZ_dR = 0.789 - 0.170 * logS
        dZ_dN = 0.726 + 0.022 * logS
        dZ_dS = 1.069 - 0.170 * logR + 0.022 * logN
        
        logOH12_err[pts_lower] = np.sqrt(
            (dZ_dR * dlogR)**2 +
            (dZ_dN * dlogN)**2 +
            (dZ_dS * dlogS)**2
        )
        
        
        # Upper branch formula
        logR = logO3S2[pts_upper]
        logN = logN2Hb[pts_upper]
        logS = logS2Hb[pts_upper]
        dlogR = logO3S2_err[pts_upper]
        dlogN = logN2Hb_err[pts_upper]
        dlogS = logS2Hb_err[pts_upper]
        
        
        # Error propagation for upper branch
        dZ_dR = 0.030 + 0.182 * logS
        dZ_dN = 0.751 + 0.508 * logS
        dZ_dS = -0.349 + 0.182 * logR + 0.508 * logN  
        
        logOH12_err[pts_upper] = np.sqrt(
            (dZ_dR * dlogR)**2 +
            (dZ_dN * dlogN)**2 +
            (dZ_dS * dlogS)**2
        )
        
        logOH12_err[~SF_region] = np.nan
        
        return logOH12, logOH12_err 
    
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
    