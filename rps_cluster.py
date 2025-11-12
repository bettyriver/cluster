#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:20:36 2025

@author: ymai0110
"""
import sys
sys.path.insert(0,'/Users/ymai0110/Documents/myPackages/metalpy/')
sys.path.insert(0,'/Users/ymai0110/Documents/MyPackages/metalpy/')
from metalgradient import log10_ratio_with_error, calculate_z_and_error
from metalgradient import ellip_distarr, pix_to_kpc, arcsec_to_kpc

from BPT import bptregion
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from pyblobby3d.b3dcomp import map_limits
import matplotlib as mpl
import os

def find_file_by_id(folder_path, target_id):
    for filename in os.listdir(folder_path):
        if filename.startswith(f"{target_id}_"):
            return filename
    return None

class cmap:
    flux = 'Oranges'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'

def plot_overview(catid,savepath=None):
    '''
    plot the following plots in the subplots
    Matt's classification, BPT map, metallicity map, metallicity vs radius

    Returns
    -------
    None.

    '''
    parent_path = '/Users/ymai0110/Documents/cluster_galaxies/'
    matt_path = parent_path + 'Matt_map/EW_MapsV0_12/v0.12/'
    oguzhan_table = pd.read_csv(parent_path+'cluster_classification_from_Oguzhan/SAMI_DR3_Cluster_Galaxies_Oguzhan_sample.csv')
    # read fits file
    file_name = find_file_by_id(matt_path,str(catid))
    fits_file = fits.open(matt_path+file_name)
    
    row = oguzhan_table[oguzhan_table['CATID'] == catid]
    
    ellip = row['ellip'].values[0]
    pa = row['PA'].values[0]
    z = row['z_spec'].values[0]
    re_arcsec = row['r_e'].values[0]
    re_kpc = arcsec_to_kpc(rad_in_arcsec=re_arcsec, z=z)
    
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(2,3,figsize=(23,10), 
                           gridspec_kw={'wspace':0.4,'hspace':0.35})
    ax = ax.ravel()
    
    classi_map = fits_file['SPEC_CLASS'].data
    
    plot_class(ax=ax[0], classi_map=classi_map)
    
    plot_bpt(catid=catid,ax_bptdiagram=ax[1],ax_bptmap=ax[2])
    
    metal_map, metal_err_map = plot_metallicity(
        ax=ax[3], fits_file=fits_file, catid=catid,
                     ellip=ellip, pa=pa, z=z, re_kpc=re_kpc)
    
    plot_metal_radius(ax=ax[4], metal_map=metal_map, 
                      metal_err_map=metal_err_map, 
                      ellip=ellip, pa=pa, z=z, re_kpc=re_kpc)
    plt.suptitle(str(catid))
    
    
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()
    
    
    

def plot_class(ax,classi_map):
    '''
    plot the spaxel classification from Matt's 2019 paper
    

    Parameters
    ----------
    ax : matplotlib.ax
        ax to plot
    classi_map: 2-d array
        2d map of spaxel classification from Matt

    Returns
    -------
    None

    '''
    
    category_names = ['PAS',
                      'rNSF',
                      'rINT',
                      'wNSF',
                      'sNSF',
                      'INT',
                      'wSF',
                      'SF',
                      'HDS',
                      'NSF_HDS']
    
    # Create a colormap and a corresponding normalization
    cmap = plt.get_cmap('tab10')
    bounds = np.arange(11) + 0.5  # For 10 discrete values: 1-10
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot with imshow
    
    img = ax.imshow(classi_map, cmap=cmap, norm=norm,origin='lower')
    
    # Optional: add colorbar with correct ticks
    
    
    cbar = plt.colorbar(img, ax=ax,ticks=np.arange(1,11))
    cbar.set_ticklabels(category_names)
    #cbar.set_label('Category')
    
    

    


def plot_bpt(catid, ax_bptdiagram=None,ax_bptmap=None):
    '''
    read the fits file from datacentral (ha,hb,nii,oiii) needed for make bpt map
    plot bpt map and/or bpt diagram

    Parameters
    ----------
    
        
    catid: int
        the id of SAMI galaxies
    ax_bptdiagram: matplotlib.ax
        ax to plot bpt diagram (x: nii/ha, y:oiii/hb)
    ax_bptmap: matplotlib.ax
        ax to plot bpt map (galaxy map color by bpt classification)

    Returns
    -------
    None.

    '''
    
    ### read data ####
    #ha = dmap['Ha_Flux_mso'].data
    #ha_err = dmap['Ha_flux_mso_err'].data
    ha, ha_err = read_emi_dc(catid=catid, emi_line='Halpha')
    ha_snr = ha/ha_err
    
    #nii = dmap['NIIR_Flux_mso'].data
    #nii_err = dmap['NIIR_flux_mso_err'].data
    
    nii, nii_err = read_emi_dc(catid=catid, emi_line='NII6583')
    nii_snr = nii/nii_err
    
    #oiii = dmap['OIIIR_mso_flux'].data
    #oiii_err = dmap['OIIIR_flux_mso_err'].data
    oiii, oiii_err = read_emi_dc(catid=catid, emi_line='OIII5007')
    oiii_snr = oiii/oiii_err
    
    #hb = dmap['Hb_Flux_mso'].data
    #hb_err = dmap['Hb_flux_mso_err'].data
    hb, hb_err = read_emi_dc(catid=catid, emi_line='Hbeta')
    hb_snr = hb/hb_err
    
    ###########################################
    #crit_mask = (ha_mask == 0) & (nii_mask == 0)&(oiii_mask == 0)&(hb_mask==0)
    crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
    crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
    indplot =  crit_err & crit_snr
    
    ############################################
    ##constrction construction coordinates###
    nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
    ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
    # ! note change indexing from 'xy' to 'ij', not sure if it's correct, but it works.
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    
    ##########caculate bpt region##################
    x = np.log10(nii/ha)
    y = np.log10(oiii/hb)
    radd = np.sqrt(xpos**2 + ypos**2)
    rad = np.median(radd)+2
    ##########################
    ##########################
    ##########################
    #########plot bpt diagram##################
    AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    x_type = np.full_like(xpos, np.nan)
    y_type = np.full_like(xpos, np.nan)
    x_type[indplot] = x[indplot]
    y_type[indplot] = y[indplot]
    AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
    
    #axs[0].tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
    if ax_bptdiagram is not None:
        ax_bptdiagram.set_title('BPT-Diagram',fontsize=25)
        ax_bptdiagram.set_xlim(-1.5,0.5)
        ax_bptdiagram.set_ylim(-1.0,1.5)
        ax_bptdiagram.plot(x_type[SF],y_type[SF],'.',color = 'blue')##SF
        ax_bptdiagram.plot(x_type[CP],y_type[CP],'.',color = 'lime')##CP
        ax_bptdiagram.plot(x_type[AGN],y_type[AGN],'.',color = 'red')##AGN
        ax_bptdiagram.set_xlabel(r"$\rm{log_{10}(NII/H\alpha)}$",fontsize=25)
        ax_bptdiagram.set_ylabel(r"$\rm{log_{10}(OIII/H\beta)}$",fontsize=25)
        x1 = np.linspace(-1.5, 0.2, 100)
        y_ke01 = 0.61/(x1-0.47)+1.19
        ax_bptdiagram.plot(x1,y_ke01)
        x2 = np.linspace(-1.5, -.2, 100)
        y_ka03 = 0.61/(x2-0.05)+1.3
        ax_bptdiagram.plot(x2,y_ka03,'--')
        ax_bptdiagram.legend(('SF','Comp','AGN','Ke 01','Ka 03'),loc='best',fontsize = 15.0,markerscale = 2)
        
    ###########################
    ########bpt diagram map#########################
    if ax_bptmap is not None:
        ax_bptmap.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
        ax_bptmap.set_title(r'resolved BPT-map',fontsize=25)
        region_type = np.full_like(xpos, np.nan)
        region_color = ['red','lime','blue']
        region_name = ['AGN', 'Comp', 'HII']
        # AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
        region_type[AGN] = 1
        region_type[CP] = 2
        region_type[SF] = 3
        new_region = np.full_like(xpos, np.nan)
        new_region[indplot] = region_type[indplot]
        bounds = [0.5, 1.5, 2.5, 3.5] # set color for imshow
        cmap = colors.ListedColormap(region_color)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        # the map data was up-down inverse as the optical image from sdss
        ax_bptmap.pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
        ax_bptmap.set_xlabel('arcsec',fontsize=25)
        ax_bptmap.set_ylabel('arcsec',fontsize=25)
    
def read_emi_dc(catid,emi_line,datapath_other=None):
    '''
    read the emission line product from datacentral cubes

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
    
    datapath = '/Users/ymai0110/Documents/cluster_galaxies/SAMI_datacentral/'
    datapath += 'sami_strong_emission/'
    if datapath_other is not None:
        datapath = datapath_other
    datapath += 'dr3/ifs/' + str(catid) + '/'
    datapath += str(catid) + '_A_' + emi_line
    datapath += '_default_recom-comp.fits'
    if not os.path.exists(datapath):
        return None, None
    fits_file = fits.open(datapath)
    emi_map = fits_file[0].data
    emi_err_map = fits_file[1].data
    
    if emi_map.ndim==3:
        return emi_map[0], emi_err_map[0]
    else:
        return emi_map, emi_err_map
    

def plot_metallicity(ax,fits_file,catid,ellip,pa,z,re_kpc):
    '''
    plot metallicity map

    Parameters
    ----------
    ax : matplotlib.ax
        ax to plot.
    fits_file:
        fits file from Matt
    catid : int
        id of SAMI galaxies

    Returns
    -------
    None.

    '''
    spec_class = fits_file['SPEC_CLASS'].data
    
    #ha_map = fits_file['Ha_Flux_mso'].data
    #ha_err_map = fits_file['Ha_flux_mso_err'].data

    #nii_map = fits_file['NIIR_Flux_mso'].data
    #nii_err_map = fits_file['NIIR_flux_mso_err'].data
    
    ha_map, ha_err_map = read_emi_dc(catid=catid, emi_line='Halpha')
    nii_map, nii_err_map = read_emi_dc(catid=catid, emi_line='NII6583')
    

    query = spec_class==8
    
    ha_map[~query] = np.nan
    nii_err_map[~query] = np.nan
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                                a_err=nii_err_map, 
                                                b_err=ha_err_map)
        
    metal_map_n2ha, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    clim = map_limits(metal_map_n2ha,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im0 = ax.imshow(metal_map_n2ha,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax,fraction=0.047)
    cb0.set_label(label='12+log(O/H)',fontsize=20)
    ax.set_title('metallicity map (n2ha)')
    
    
    map_shape = ha_map.shape
    dist_arr = ellip_distarr(size=map_shape, 
                             centre=(map_shape[1]/2-0.5,map_shape[0]/2-0.5),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
    radius_re = radius_kpc/re_kpc
    
    ax.contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    
    linestylelist=['dashed']
    colorlist=['g']
    label_column=['1 R$_\mathrm{e}$']
    columns = [ax.plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(len(colorlist))]

    ax.legend( columns,  label_column,loc='lower right', prop={'size': 12})
    
    return metal_map_n2ha, metal_err_n2ha

def get_n2ha_metal(fits_file,catid,ellip,pa,z):
    '''
    get n2ha metal map and radius_kpc map

    Parameters
    ----------
    fits_file:
        fits file from Matt
    catid : int
        id of SAMI galaxies

    Returns
    -------
    None.

    '''
    spec_class = fits_file['SPEC_CLASS'].data
    
    #ha_map = fits_file['Ha_Flux_mso'].data
    #ha_err_map = fits_file['Ha_flux_mso_err'].data

    #nii_map = fits_file['NIIR_Flux_mso'].data
    #nii_err_map = fits_file['NIIR_flux_mso_err'].data
    
    ha_map, ha_err_map = read_emi_dc(catid=catid, emi_line='Halpha')
    nii_map, nii_err_map = read_emi_dc(catid=catid, emi_line='NII6583')
    

    query = spec_class==8
    
    ha_map[~query] = np.nan
    nii_err_map[~query] = np.nan
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                                a_err=nii_err_map, 
                                                b_err=ha_err_map)
        
    metal_map_n2ha, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
    
    
    map_shape = ha_map.shape
    dist_arr = ellip_distarr(size=map_shape, 
                             centre=(map_shape[1]/2-0.5,map_shape[0]/2-0.5),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
    
    
    
    return metal_map_n2ha, metal_err_n2ha, radius_kpc

    
def plot_metal_radius(ax,metal_map,metal_err_map,ellip,pa,z,re_kpc,R='N2Ha'):
    map_shape = metal_map.shape
    dist_arr = ellip_distarr(size=map_shape, 
                             centre=(map_shape[1]/2-0.5,map_shape[0]/2-0.5),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
    ax.errorbar(radius_kpc.ravel(),metal_map.ravel(),yerr=metal_err_map.ravel(),
                linestyle='None',marker='o',
                color='chocolate')
    ax.set_xlabel('radius [kpc]')
    ax.set_ylabel('12+log(O/H)')
    ax.axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    if len(radius_kpc[~np.isnan(metal_map)])!=0:
        r_max = np.nanmax(radius_kpc[~np.isnan(metal_map)])
        ax.set_xlim(0,r_max)

    if R=='N2O2':
        ax.axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
    elif R=='N2Ha':
        ax.axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')

def get_bin_metal(metal_map,metal_err_map,radius_map,
                  bin_size,weight_mode='no'):
    if len(metal_map)==0:
        return np.array([]), np.array([]), np.array([])
    
    non_nan_metal = radius_map[~np.isnan(metal_map)]
    if len(non_nan_metal)==0:
        return np.array([]), np.array([]), np.array([])
    
    # get the max radius where data available
    radius_max_kpc = np.nanmax(radius_map[~np.isnan(metal_map)])
    r_set = int(radius_max_kpc/bin_size)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    metal_mean = []
    metal_inv_var = []
    
    metal_mean_err = []
    metal_inv_var_err = []
    
    radius_array_new = []
    for r in radius_array:
        query = (radius_map>=r*bin_size)&(radius_map<(r+1)*bin_size)
        query_all = (radius_map>=r*bin_size)&(radius_map<(r+1)*bin_size)&(~np.isnan(metal_map))&(~np.isnan(metal_err_map))
        
        if len(metal_map[query_all]) <1:
            metal_mean.append(np.nan)
            
            metal_inv_var.append(np.nan)
            
            metal_mean_err.append(np.nan)
            metal_inv_var_err.append(np.nan)
            radius_array_new.append(r)
            
            continue
        
        
        
        
        # average 1
        metal_query = metal_map[query_all]
        metal_err_query = metal_err_map[query_all]
        avg_err = np.sqrt(np.sum(metal_err_query**2)) / len(metal_err_query)
        
        
        
        
        
        # average 2
        weights = 1 / metal_err_query**2
        ave_inv_var = np.sum(metal_query * weights) / np.sum(weights)
        ave_inv_var_err = np.sqrt(1 / np.sum(weights))
        
        # don't save if error too large or metallicity below the valid range
        if ave_inv_var<7.63 or ave_inv_var_err>0.5:
            metal_mean.append(np.nan)
            metal_inv_var.append(np.nan)
            
            metal_mean_err.append(np.nan)
            metal_inv_var_err.append(np.nan)
            radius_array_new.append(r)
            
            continue
        
        
        metal_mean.append(np.nanmean(metal_query))
        metal_mean_err.append(avg_err)
        
        metal_inv_var.append(ave_inv_var)
        metal_inv_var_err.append(ave_inv_var_err)
        
        
        
        radius_array_new.append(r)
    
    metal_mean = np.array(metal_mean)
    metal_inv_var = np.array(metal_inv_var)
    
    metal_mean_err = np.array(metal_mean_err)
    metal_inv_var_err = np.array(metal_inv_var_err)
    
    # the radius should be the center of bin, which is 0.5, 1.5, 2.5 ...
    radius_array_new = (np.array(radius_array_new) + 0.5) * bin_size
    
    if weight_mode=='no':
        return metal_mean, metal_mean_err, radius_array_new
    if weight_mode=='inv_var':
        return metal_inv_var, metal_inv_var_err, radius_array_new
    
    
    
    