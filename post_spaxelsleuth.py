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
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

def make_ss_overviewplot(metal_fits,sfr_fits,dist_arr,bin_size,re_kpc,title,
                         savepath=None):
    '''
    BPT map (contour), SFR map (contour)
    19 metallicity map + metallicity vs gradient (data points + bin data)
    
    4*10
    
    
    Returns
    -------
    None.

    '''
    
    BPT_map = sfr_fits['BPT (numeric) (total)'].data
    radius_re = dist_arr/re_kpc
    SFR_map = sfr_fits['SFR (total)'].data
    
    # check if any SF spaxel, if not, simply return 0
    has_SF = np.any(BPT_map == 0)
    
    if has_SF==False:
        print('no SF spaxel in this galaxy')
        return None
    
    
    fig,ax = plt.subplots(nrows=10,ncols=4,figsize=(20,28), 
                         gridspec_kw={'wspace': 0.45, 'hspace': 0.45})
    ax = ax.flatten()
    
    # BPT map
    plot_ax_bpt(ax[0], BPT_map,radius_re)
    # SFR map
    plot_ax_SFR(fig,ax[1],SFR_map)
    
    all_diag = hss.Spaxelsleuth.metal_ext_names
    
    SF_query = BPT_map==0.0
    
    for i, metal_diag in enumerate(all_diag):
        
        # ext name for this diagnostic
        diag_name = metal_diag[15:-9]
        
        err_16 = metal_diag[:-7] + 'error (lower) ' + metal_diag[-7:]
        err_84 = metal_diag[:-7] + 'error (upper) ' + metal_diag[-7:]
        
        err_16_map = metal_fits[err_16].data
        err_84_map = metal_fits[err_84].data
    
        err_map = (err_16_map + err_84_map)/2
        
        metal_map = metal_fits[metal_diag].data
        
        
        
        
        
        metal_map_mask = metal_map[SF_query]
        error_map_mask = err_map[SF_query]
        dist_arr_mask = dist_arr[SF_query]
        
        dist_max  = np.nanmax(dist_arr_mask)
        
        #    no SF pixel    , no metal measurement in SF pixel for this diagnostic
        if np.isnan(dist_max) or np.isnan(np.nansum(metal_map_mask)):
            continue
        
        plot_ax_metalmap(fig=fig,ax=ax[2*i+2], metal_map=metal_map,title=diag_name)
        plot_ax_metalgradient(ax=ax[2*i+3], metal_map=metal_map, 
                              metal_err_map=err_map, BPT_map=BPT_map,
                                  dist_arr=dist_arr, bin_size=bin_size,
                                  title=diag_name)
        
    plt.suptitle(title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    plt.show()
    return 1






def plot_ax_metalgradient(ax, metal_map, metal_err_map, BPT_map,
                          dist_arr, bin_size,title):
    '''
    plot the metallicity for all spaxels (grey) and bin metallicity

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    metal_map : TYPE
        DESCRIPTION.
    metal_err_map : 2d-array
        the map of metallicity error
    BPT_map: 2d-array
        BPT map
    dist_arr : 2d-array
        distance to galaxy centre, in kpc.
    bin_size : float
        bin size in kpc.
    title : str
        title of plot, diagnostic name

    Returns
    -------
    None.

    '''
    all_nan = np.isnan(metal_map).all()
    if all_nan:
        ax.set_visible(False)
        return None
    
    #ax.scatter(dist_arr.ravel(), metal_map.ravel(),c='grey',zorder=1)
    ax.errorbar(x=dist_arr.ravel(),y=metal_map.ravel(),
                yerr=metal_err_map.ravel(),fmt='o',linestyle='none',c='grey',zorder=1)
    
    # create bin dist arr for the whole table
    SF_query = BPT_map==0.0
    dist_arr_mask = dist_arr[SF_query]
    dist_max  = np.nanmax(dist_arr_mask)
    bin_num = ceil(dist_max/bin_size)
    bin_arr_kpc = []
    for i in range(bin_num):
        bin_arr_kpc.append(i*bin_size+0.5*bin_size)
    bin_arr_kpc = np.array(bin_arr_kpc)
    
    ### mean
    metal_mean_arr,metal_mean_err_arr,metal_inv_ave_arr,metal_inv_ave_err_arr = \
        get_bin_metallicity(metal_map=metal_map,
                            error_map=metal_err_map,BPT_map=BPT_map,
                            dist_arr=dist_arr,bin_size=bin_size)
    
    query_1 = np.isnan(metal_mean_arr) | np.isnan(metal_mean_err_arr)
    
    count = np.count_nonzero(~np.isnan(metal_mean_arr[~query_1]))
    
    if count<3:
        return None
    
    popt_mean, pcov_mean = curve_fit(linear_func, bin_arr_kpc[~query_1], 
                                     metal_mean_arr[~query_1],
                                     sigma=metal_mean_err_arr[~query_1])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))


    ax.errorbar(bin_arr_kpc, metal_mean_arr,
                    yerr=metal_mean_err_arr,fmt='o',linestyle='none',c='b',zorder=5)
    
    plot_radius_array = bin_arr_kpc - 0.5*bin_size
    plot_radius_array = np.concatenate((plot_radius_array,
                                        [plot_radius_array[-1]+1*bin_size]))
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_mean), 
                color='b', linestyle='--',zorder=6,label='unweighted')
    
    ### inverse variance
    
    query_3 = np.isnan(metal_inv_ave_arr) | np.isnan(metal_inv_ave_err_arr)
    
    popt_inv, pcov_inv = curve_fit(linear_func, bin_arr_kpc[~query_3],
                                   metal_inv_ave_arr[~query_3],
                                   sigma=metal_inv_ave_err_arr[~query_3])
    
    a_inv, b_inv = popt_inv
    a_inv_err, b_inv_err = np.sqrt(np.diag(pcov_inv))
    
    ax.errorbar(bin_arr_kpc+0.05,metal_inv_ave_arr,yerr=metal_inv_ave_err_arr,
                fmt='o',linestyle='none',
                c='r',zorder=5)
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_inv), 
                color='r', linestyle='--',zorder=6,label='inv variance')
    
    ax.text(0.5, 0.35, 'y=({:.3f}$\pm${:.3f})x + ({:.3f}$\pm${:.3f})'.format(a_inv,a_inv_err,b_inv,b_inv_err), 
            horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,c='darkorange',fontsize=10,
         bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.2'))
    
    ax.set_xlim(left=0)
    ax.legend(prop={'size': 5})
    ax.set_xlabel('radius [kpc]')
    ax.set_ylabel('12+log(O/H)')
    ax.set_title(title)
    

def calculate_gradient(radius_arr,metal_arr, metal_err_arr):
    '''
    calculate the metallicity gradient and central metallicity.
    if the number of bin is less then 4, return nan and nan

    Parameters
    ----------
    radius_arr : TYPE
        DESCRIPTION.
    metal_arr : TYPE
        DESCRIPTION.
    metal_err_array : TYPE
        DESCRIPTION.

    Returns
    -------
    gradient: float
            the metallicity gradient in dex/kpc
    gradient_err: float
            the error of gradient
    central_metal: float
            the central metallity derived through fitting
    central_metal_err: float
            the error of central metallcity

    '''
    
    count = np.count_nonzero((~np.isnan(metal_arr))&(~np.isnan(metal_err_arr)))
    if count <=3 : 
        gradient = np.nan
        gradient_err = np.nan
        central_metal = np.nan
        central_metal_err = np.nan
        return gradient,gradient_err, central_metal,central_metal_err
    
    # more than 4 valid bin
    ### inverse variance
    
    query = np.isnan(metal_arr) | np.isnan(metal_err_arr)
    
    popt, pcov = curve_fit(linear_func, radius_arr[~query],
                                   metal_arr[~query],
                                   sigma=metal_err_arr[~query])
    
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    gradient = a
    gradient_err = a_err 
    central_metal = b
    central_metal_err = b_err
    
    return gradient,gradient_err, central_metal,central_metal_err
    


def linear_func(x, a, b):
    return a * x + b


def plot_ax_metalmap(fig,ax, metal_map,title):
    '''
    plot metallicity map

    Parameters
    ----------
    fig : 
    ax : TYPE
        DESCRIPTION.
    metal_map : TYPE
        DESCRIPTION.
    title : str
        diagnostic name

    Returns
    -------
    None.

    '''
    
    all_nan = np.isnan(metal_map).all()
    if all_nan:
        ax.set_visible(False)
        return None
    
    clim = map_limits(metal_map,pct=95)
    
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im = ax.imshow(
        metal_map,
        origin='lower',
        interpolation='nearest',
        norm=norm,
        cmap='magma'
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(O/H)+12',fontsize=10)
    ax.set_title(title)

def plot_ax_SFR(fig,ax,SFR_map):
    
    clim = map_limits(np.log10(SFR_map),pct=95)
    
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    im = ax.imshow(
        np.log10(SFR_map),
        origin='lower',
        interpolation='nearest',
        norm=norm,
        cmap='OrRd'
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(SFR)", fontsize=10)
    ax.set_title('SFR-map')


def map_limits(data, pct=100.0, vlim=None, absolute=False):
    """
    Get limits for maps.

    Parameters
    ----------
    data : 2D numpy.array or list of 2D numpy.array
        Map data or list of map data
    pct : float, default 95%
        Percentile to use to calculate limit.
    vlim : list of float, default None
        Absolute limits applied to map.
    absolute : bool, default False
        Whether to calculate limits about 0.
    """
    if isinstance(data, list):
        data_fin = []
        for d in data:
            data_fin.append(d[np.isfinite(d)])
        data_fin = np.array(data_fin)
    else:
        data_fin = data[np.isfinite(data)]

    if vlim is not None:
        data_lim = vlim
    elif not absolute:
        data_lim = [
                np.nanpercentile(data_fin, 100.0 - pct),
                np.nanpercentile(data_fin, pct)
                ]
    else:
        data_abs = np.absolute(data)
        data_lim_max = np.nanpercentile(data_abs, pct)
        data_lim = [-data_lim_max, data_lim_max]

    return data_lim    


def plot_ax_bpt(ax,BPT_map,radius_re):
    '''
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    BPT_map : TYPE
        DESCRIPTION.
    radius_re : 2d-array
    distance to center normalise by re, i.e. radius_kpc/re_kpc

    Returns
    -------
    None.

    '''
    # -1 is no classified
    category_names = ['no class',#-1
                      'SF',#0
                      'Composite',#1
                      'LINER',#2
                      'Seyfert',#3
                      'Ambigous']#4
    
    # Create a colormap and a corresponding normalization
    cmap = plt.get_cmap('tab10')
    bounds = np.arange(6) - 1.5  # For 5 discrete values: 0-4
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot with imshow
    
    img = ax.imshow(BPT_map, cmap=cmap, norm=norm,origin='lower')
    
    # Optional: add colorbar with correct ticks
    
    
    cbar = plt.colorbar(img, ax=ax,ticks=np.arange(-1,5))
    cbar.set_ticklabels(category_names)
    
    ax.contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid',label='0.5 R$_\mathrm{e}$')
    ax.contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    ax.contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted',label='1.5 R$_\mathrm{e}$')
    
    linestylelist=['solid','dashed','dotted']
    colorlist=['r','g','b']
    label_column=['0.5 R$_\mathrm{e}$','1 R$_\mathrm{e}$','1.5 R$_\mathrm{e}$']
    columns = [ax.plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(3)]

    ax.legend( columns,  label_column,loc='lower right', prop={'size': 5})
    ax.set_title('BPT-map')
    
    

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
        ave_inv_var = np.nansum(metal_map_mask_query * weights) / np.nansum(weights)
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
        2d-array that gives the distance to the centre in kpc
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
    
    if has_SF==False:
        print('no SF spaxel in this galaxy')
        return None
    
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


def get_overview_bulk(galaxy_list,bin_type,savepath):
    '''
    

    Parameters
    ----------
    galaxy_list : TYPE
        DESCRIPTION.
    bin_type : TYPE
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    parent_path = '/Users/ymai0110/Documents/cluster_galaxies/'
    
    spaxelsleuth_path = parent_path + 'spaxelsleuth/'
    metal_fits_path =  spaxelsleuth_path + 'metallicity/'
    sfr_path = spaxelsleuth_path + 'sfr_related/'
    
    cluster_csv = pd.read_csv(parent_path+
                              'cluster_classification_from_Oguzhan/'+
                              'HECTOR_cluster_galaxies_updated.csv')
    
    
    for galaxy_id in galaxy_list:
        metal_fits = fits.open(metal_fits_path + galaxy_id + '_metalicity.fits')
        sfr_fits = fits.open(sfr_path + galaxy_id + '_sfr_related.fits')
        # create dist_arr in kpc for this galaxy
        query = cluster_csv['name']==galaxy_id[:-1]
        
        map_shape = sfr_fits['BPT (numeric) (total)'].data.shape
        
        xcen = cluster_csv['xcen'][query].to_numpy()[0] # cube center and galaxy center, in the unit of pixel
        ycen = cluster_csv['ycen'][query].to_numpy()[0]
        z = cluster_csv['z'][query].to_numpy()[0]
        re = cluster_csv['Re'][query].to_numpy()[0] # r-band major axis effective radius in arcsec
        b_a = cluster_csv['B_on_A'][query].to_numpy()[0]
        pa = cluster_csv['PA'][query].to_numpy()[0] # r-band position angle in deg
        fwhm = cluster_csv['fwhm'][query].to_numpy()[0] # FWHM of PSF in cube in arcsec
        ellip = 1 - b_a # r-band ellipticity
        
        # for galaxies b_a=1, pa is nan, need to give it a value
        if b_a == 1: 
            pa = 0
        
        dist_arr = ellip_distarr(size=map_shape, 
                                 centre=(xcen,ycen),
                                 ellip=ellip, pa=pa*np.pi/180,angle_type='NTE')
        radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
        
        re_kpc = arcsec_to_kpc(rad_in_arcsec=re, z=z)
        fwhm_kpc = arcsec_to_kpc(rad_in_arcsec=fwhm, z=z)
        # calculate the bin size in kpc depends on the bin type
        if bin_type=='Re':
            bin_size_kpc = re_kpc/4.0
        elif bin_type=='PSF':
            bin_size_kpc = fwhm_kpc/2.0
        
        # calculate the dataframe for bin metallicity table and save it
        
        overview = make_ss_overviewplot(metal_fits=metal_fits,
                                         sfr_fits=sfr_fits,
                                         dist_arr=radius_kpc,
                                         bin_size=bin_size_kpc,
                                         re_kpc=re_kpc,title=galaxy_id,
                                 savepath=savepath+galaxy_id[:-1]+'.png')
        
        if overview is None:
            
            print(galaxy_id+' no measurement.')
    
    

def get_table_bulk(galaxy_list,bin_type,savepath):
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
    
    cluster_csv = pd.read_csv(parent_path+
                              'cluster_classification_from_Oguzhan/'+
                              'HECTOR_cluster_galaxies_updated.csv')
    
    
    for galaxy_id in galaxy_list:
        metal_fits = fits.open(metal_fits_path + galaxy_id + '_metalicity.fits')
        sfr_fits = fits.open(sfr_path + galaxy_id + '_sfr_related.fits')
        # create dist_arr in kpc for this galaxy
        query = cluster_csv['name']==galaxy_id[:-1]
        
        map_shape = sfr_fits['BPT (numeric) (total)'].data.shape
        
        xcen = cluster_csv['xcen'][query].to_numpy()[0] # cube center and galaxy center, in the unit of pixel
        ycen = cluster_csv['ycen'][query].to_numpy()[0]
        z = cluster_csv['z'][query].to_numpy()[0]
        re = cluster_csv['Re'][query].to_numpy()[0] # r-band major axis effective radius in arcsec
        b_a = cluster_csv['B_on_A'][query].to_numpy()[0]
        pa = cluster_csv['PA'][query].to_numpy()[0] # r-band position angle in deg
        fwhm = cluster_csv['fwhm'][query].to_numpy()[0] # FWHM of PSF in cube in arcsec
        ellip = 1 - b_a # r-band ellipticity
        
        # for galaxies b_a=1, pa is nan, need to give it a value
        if b_a == 1: 
            pa = 0
        
        dist_arr = ellip_distarr(size=map_shape, 
                                 centre=(xcen,ycen),
                                 ellip=ellip, pa=pa*np.pi/180,angle_type='NTE')
        radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z,CD2=0.0001388888)
        
        re_kpc = arcsec_to_kpc(rad_in_arcsec=re, z=z)
        fwhm_kpc = arcsec_to_kpc(rad_in_arcsec=fwhm, z=z)
        # calculate the bin size in kpc depends on the bin type
        if bin_type=='Re':
            bin_size_kpc = re_kpc/4.0
        elif bin_type=='PSF':
            bin_size_kpc = fwhm_kpc/2.0
        
        # calculate the dataframe for bin metallicity table and save it
        
        df = make_bin_metal_table(metal_fits=metal_fits,
                                  sfr_fits=sfr_fits,
                                  dist_arr=radius_kpc,
                                  bin_size=bin_size_kpc)
        if isinstance(df, pd.DataFrame):
            
        
            df.to_csv(savepath+galaxy_id[:-1]+'.csv')
        else:
            print(galaxy_id+' no measurement.')


##Create an elliptical distance array for deprojected radial profiles.
#size=the size of the array. Should be the same as an image input for uf.radial_profile
#centre=centre of the ellipse
#ellip=ellipticity of the ellipse=1-b/a
#pa=position angle of the ellipse starting along the positive x axis of the image
#Angle type: 'NTE' = North-Through-East (Default). 'WTN'=West-Through-North
def ellip_distarr(size,centre,ellip,pa,scale=None, angle_type='NTE'):
    '''
    1. If PA means the angle of major axis with respect to the North (e.g. 
        PA in SAMI and profound.ang in MAGPI. note the ang in MAGPI is
        in the unit of deg, need to convert to rad), use angle_type='WTN'.
    2. If PA means the angle of major axis with respect to the West, i.e. the
        positive x axis of the image, use angle_type='NTE'.

    '''
    y,x=np.indices(size)
    x=x-centre[0]
    y=y-centre[1]
    r=np.sqrt(x**2 + y**2)
    theta=np.zeros(size)
    theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
    theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
    theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
    theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])
    if angle_type=='NTE':
        theta=theta+np.pi/2.0
    scdistarr=np.nan_to_num(np.sqrt((((np.sin(theta-pa))**2)+(1-ellip)**-2*(np.cos(theta-pa))**2))*r) ##SHOULD BE (1-ellip)**-2 !!!!
    #if scale=None:
    return scdistarr

def pix_to_kpc(radius_in_pix,z,CD2=5.55555555555556e-05):
    '''
    author: yifan

    Parameters
    ----------
    radius_in_pix : float
        radius get from cont_r50_ind=get_x(r,cog,0.5)
    z : float
        redshift
    CD2 : float
        CD2 in fits

    Returns
    -------
    None.

    '''
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    ang = radius_in_pix * CD2 # deg
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance
    radius_in_kpc = ang*np.pi/180*distance*1000
    return radius_in_kpc

def arcsec_to_kpc(rad_in_arcsec,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
    return rad_in_kpc