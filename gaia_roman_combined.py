import numpy as np
import pandas as pd
import os

import astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

datapath = './data/'

allowed_filters = ['F062','F087','F106','F129','F146','F158','F184','F213'] #allowed Roman filters
roman_pix_scale = 107.8577405 #mas/pixel

def roman_position_precision(mags_ab,filt,
                             datapath = datapath):
    '''
    For a given AB magnitude and Roman filter,
    return the expected position uncertainty of a star.
    The underlying uncertaintyies come from running the 
    accompanying roman_astrometric_precision.ipynb notebook, which 
    uses pandeia and stpsf to estimate Roman position measurements.

    Inputs:
    mags_ab = numpy array of AB magnitudes to calculate position uncertainties
    filt = one of ['F062','F087','F106','F129',
                   'F146','F158','F184','F213'],  Roman filter

    Outputs:
    sigma_xy = numpy array of position uncertainty of source in Roman image (mas)    
    '''

    if filt not in allowed_filters:
        raise ValueError(f'ERROR: Chosen filter {filt} is not in the allowed filter list. Please try again.')

    fname = f'{datapath}roman_{filt}_pos_errs.csv'
    if not os.path.isfile(fname):
        raise ValueError(f'ERROR: Could not find position error csv for filter {filt}.'\
                         +f' Please download the appropriate data or run the roman_astrometric_precision.ipynb first.')
    input_data = pd.read_csv(fname)
    
    return np.interp(mags_ab,input_data['mags_ab'],input_data['pos_errs_mas'],left=np.nan,right=np.nan)


def gaia_astrometry_precision(gmag,
                              era = 'DR4',
                              gmax_pms = 20.7,
                              gmax_pos = 21.5,
                              gmin = 3):
    '''
    For a given Gaia G (gmag), returns the position, parallax, PM 
    precision (mas) for a given era (DR4, DR5)
    from https://www.cosmos.esa.int/web/gaia/science-performance

    Inputs:
    gmags = numpy array of Gaia G mags to calculate astrometric uncertainties
    era = one of ['DR3','DR4','DR5'], data release of Gaia to consider
    gmax_pms = 20.7 #max mag where parallaxes/PMs are measured
    gmax_pos = 21.5 #max mag where Positions are measured
    gmin = 3 #min mag where measurements are made

    Outputs:
    sigma_dracosdec = numpy array of RA position uncertainties (mas)
    sigma_ddec = numpy array of Dec position uncertainties (mas)
    sigma_pmracosdec = numpy array of PMRA uncertainties (mas/yr)
    sigma_pmdec = numpy array of PMDec uncertainties (mas/yr)
    sigma_parallax = numpy array of parallax uncertainties (mas)
    '''
    
    z = np.maximum(10**((0.4) * (13 - 15)), np.power(10,0.4 * (gmag - 15)))
    
    if era == 'DR4':
        T_factor = 0.749
        parallax_to_dracosdec_mult = 0.8
        parallax_to_ddec_mult = 0.7
        parallax_to_pmracosdec_mult = 0.58
        parallax_to_pmdec_mult = 0.50
    elif era == 'DR5':
        T_factor = 0.527
        parallax_to_dracosdec_mult = 0.8
        parallax_to_ddec_mult = 0.7
        parallax_to_pmracosdec_mult = 0.29
        parallax_to_pmdec_mult = 0.25
    elif era == 'DR3':
        T_factor = 1.0
        parallax_to_dracosdec_mult = 0.8
        parallax_to_ddec_mult = 0.7
        parallax_to_pmracosdec_mult = 1.03
        parallax_to_pmdec_mult = 0.89
    else:
        raise ValueError(f'Invalid era of {era}. Please choose DR3, DR4, DR5')
    sigma_parallax = T_factor * np.sqrt(40 + 800 * z + 30 * np.power(z,2))/1000

    sigma_dracosdec = sigma_parallax*parallax_to_dracosdec_mult
    sigma_ddec = sigma_parallax*parallax_to_ddec_mult
    sigma_pmracosdec = sigma_parallax*parallax_to_pmracosdec_mult
    sigma_pmdec = sigma_parallax*parallax_to_pmdec_mult

    sigma_dracosdec[gmag > gmax_pos] = np.nan
    sigma_ddec[gmag > gmax_pos] = np.nan
    sigma_parallax[gmag > gmax_pms] = np.nan
    sigma_pmracosdec[gmag > gmax_pms] = np.nan
    sigma_pmdec[gmag > gmax_pms] = np.nan

    sigma_dracosdec[gmag < gmin] = np.nan
    sigma_ddec[gmag < gmin] = np.nan
    sigma_parallax[gmag < gmin] = np.nan
    sigma_pmracosdec[gmag < gmin] = np.nan
    sigma_pmdec[gmag < gmin] = np.nan

    return sigma_dracosdec,sigma_ddec,sigma_pmracosdec,sigma_pmdec,sigma_parallax

def delta_ra_dec_per_parallax_VECTORIZED(other_times,gaia_time,ra,dec):
    '''
    Calculate the parallax offset vectors at other_times for a position (ra,dec)
    '''
    
    #THIS CODE IS REPURPOSED FROM CODE FROM MELODIE KAO
    
    #choose any parallax because we will scale by it later
    parallax = 1.0*u.mas
    distance = (1/parallax.value)*u.kpc
    delta_time = (gaia_time-other_times).to(u.year).value
    dates = other_times[:,None]+(np.array([delta_time*0,delta_time]).T)*u.year
    dates = Time(dates, format='mjd')
    
    sun_loc = astropy.coordinates.get_sun(dates)
    
    sun_skycoord = SkyCoord(frame='gcrs', obstime=dates,
                            ra = sun_loc.ra, dec = sun_loc.dec)
    #SHOULD BE REFERING TO ROMAN'S TRUE ECLIPTIC AT SOME POINT
    sun_eclon = sun_skycoord.geocentrictrueecliptic.lon
#    sun_eclat = sun_skycoord.geocentrictrueecliptic.lat

    T = ((dates-Time('2000.0', format='jyear')).to(u.year).value)/100
    
    ecc = (23+26/60+21.406/3600)-46.836769/3600*T-0.0001831/3600*np.power(T,2)\
            + 0.00200340/3600*np.power(T,3)-0.576e-6*np.power(T,4)-4.34e-8/3600*np.power(T,5)

    #SHOULD CHANGE TO ROMAN's L2 OBLIQUITY AT SOME POINT!
    ecc = ecc[:,0]*np.pi/180 #earth's obliquity in radians
    sun_eclon = (sun_eclon.value*np.pi/180)[:,0]
    cos_alpha = np.cos(ra*np.pi/180)
    sin_alpha = np.sin(ra*np.pi/180)
    cos_delta = np.cos(dec*np.pi/180)
    sin_delta = np.sin(dec*np.pi/180)
    cos_lam = np.cos(sun_eclon)
    sin_lam = np.sin(sun_eclon)
    cos_ecc = np.cos(ecc)
    sin_ecc = np.sin(ecc)
    
    dalpha = cos_alpha*cos_ecc*sin_lam-sin_alpha*cos_lam
    ddelta = cos_delta*sin_ecc*sin_lam-cos_alpha*sin_delta*cos_lam\
                -sin_alpha*sin_delta*cos_ecc*sin_lam
    parallax_vectors = np.zeros((len(other_times),2))
    parallax_vectors[:,0] = dalpha
    parallax_vectors[:,1] = ddelta
    
    return parallax_vectors


class gaia_roman_astrometric_precision:
    """
    takes lists of Roman magnitudes (and corresponding filter names), Gaia magnitudes
    to be used in determining the astrometric improvement from combining telescopes

    provide ra,dec coordinates (in degrees) to properly account for parallax in the 
    calculations, otherwise parallax will be ignored and only positions and PM 
    precisions will be updated.

    Inputs:
    roman_filters = list of filters used, shape = (n_filters)
    roman_mags = numpy array, roman magnitudes in different filters of all stars, shape = (n_stars,n_filters)
    gaia_mags = numpy array, gaia magnitudes, shape = (n_stars)
    observation_list = description of observations, shape = (n_observations,(epoch_MJD,roman_filter,N_images_at_epoch))
    roman_pos_floor_err = uncertainty floor (in mas) of Roman position measurements
    ra,dec = target coordinate in degrees
    """
    
    def __init__(self, roman_mags, roman_filters, gaia_mags, observation_list,
                 gaia_era = 'DR4', roman_pos_floor_err = 0.01*roman_pix_scale, 
                 gaia_ref_epoch = 2016.0, ra=None, dec=None):

        """
        for all the given roman magnitudes, save the corresponding position uncertainties (mas),
        as well as Gaia-based astrometric precision covariance matrices given the Gaia mags

        then use the observation information to update the astrometry uncertainties
        """

        #general information
        self.n_stars = len(gaia_mags)
        self.n_epochs = len(observation_list)

        #roman information
        self.n_filters = len(roman_filters)
        self.roman_filters = np.array(roman_filters)
        self.roman_mags = roman_mags
        self.roman_pos_errs = np.zeros_like(roman_mags)
        
        self.roman_covs = np.zeros((self.n_stars,self.n_filters,2,2))
        
        for filt_ind,filt in enumerate(roman_filters):
            self.roman_pos_errs[:,filt_ind] = roman_position_precision(self.roman_mags[:,filt_ind],filt)
            self.roman_covs[:,filt_ind,0,0] = np.power(self.roman_pos_errs[:,filt_ind],2)
            self.roman_covs[:,filt_ind,1,1] = np.power(self.roman_pos_errs[:,filt_ind],2)
        self.good_roman_errs = np.isfinite(self.roman_pos_errs)
        self.roman_covs += np.eye(2)*roman_pos_floor_err**2

        self.roman_inv_covs = np.linalg.inv(self.roman_covs)        
        self.roman_inv_covs[~self.good_roman_errs] = 0        

        #gaia information
        self.gaia_time = Time(gaia_ref_epoch,format='jyear',scale='tcb')
        self.gaia_era = gaia_era
        self.gaia_gs = gaia_mags
        self.gaia_precisions = gaia_astrometry_precision(self.gaia_gs,era=gaia_era)
        self.gaia_dracosdec_errs = self.gaia_precisions[0]
        self.gaia_ddec_errs = self.gaia_precisions[1]
        self.gaia_pmra_errs = self.gaia_precisions[2]
        self.gaia_pmdec_errs = self.gaia_precisions[3]
        self.gaia_parallax_errs = self.gaia_precisions[4]

        self.good_gaia_pos = np.isfinite(self.gaia_dracosdec_errs)
        self.good_gaia_pms = np.isfinite(self.gaia_pmra_errs)
        self.missing_gaia_pms = self.good_gaia_pos & (~self.good_gaia_pms)
        
        self.gaia_inv_covs = np.zeros((self.n_stars,5,5))
        self.gaia_inv_covs[:,0,0] = np.power(self.gaia_dracosdec_errs,-2)
        self.gaia_inv_covs[:,1,1] = np.power(self.gaia_ddec_errs,-2)
        self.gaia_inv_covs[:,2,2] = np.power(self.gaia_pmra_errs,-2)
        self.gaia_inv_covs[:,3,3] = np.power(self.gaia_pmdec_errs,-2)
        self.gaia_inv_covs[:,4,4] = np.power(self.gaia_parallax_errs,-2)
        self.gaia_covs = np.zeros((self.n_stars,5,5))
        self.gaia_covs[:,0,0] = np.power(self.gaia_dracosdec_errs,2)
        self.gaia_covs[:,1,1] = np.power(self.gaia_ddec_errs,2)
        self.gaia_covs[:,2,2] = np.power(self.gaia_pmra_errs,2)
        self.gaia_covs[:,3,3] = np.power(self.gaia_pmdec_errs,2)
        self.gaia_covs[:,4,4] = np.power(self.gaia_parallax_errs,2)

        self.gaia_inv_covs[~self.good_gaia_pos] = 0
        self.gaia_covs[~self.good_gaia_pos] = np.nan
        
        self.gaia_inv_covs[self.missing_gaia_pms,2:,2:] = 0
        self.gaia_covs[self.missing_gaia_pms,2:,2:] = np.nan

        #use determinant definition of PM and Position error sizes
        self.gaia_pos_covs = self.gaia_covs[:,:2,:2]
        self.gaia_pm_covs = self.gaia_covs[:,2:4,2:4]
        self.gaia_pos_errs = np.power(np.linalg.det(self.gaia_pos_covs),1/4)
        self.gaia_pm_errs = np.power(np.linalg.det(self.gaia_pm_covs),1/4)

        #observation information
        self.obs_roman_filt_inds = np.zeros(self.n_epochs).astype(int)
        self.obs_epoch_mjds = np.zeros(self.n_epochs).astype(float)
        self.obs_n_images_per_epoch = np.zeros(self.n_epochs).astype(int)
        self.obs_roman_filt_inds = np.zeros(self.n_epochs).astype(int)
        for obs_ind in range(self.n_epochs):
            self.obs_epoch_mjds[obs_ind] = observation_list[obs_ind][0]
            self.obs_n_images_per_epoch[obs_ind] = observation_list[obs_ind][2]
            self.obs_roman_filt_inds[obs_ind] = np.where(self.roman_filters == observation_list[obs_ind][1])[0][0]

        self.n_min_epochs_indv = np.sum(self.good_roman_errs[:,self.obs_roman_filt_inds],axis=1)
        self.n_min_epochs_indv[self.good_gaia_pos] += 1
        self.n_min_epochs_indv[self.good_gaia_pms] += 1
        self.multi_epoch_stars = (self.n_min_epochs_indv > 1)
        
        self.obs_times = Time(self.obs_epoch_mjds, format='mjd')
        self.obs_dyears = (self.obs_times-self.gaia_time).to(u.year).value
        if (ra is None) or (dec is None):
            self.parallax_vectors = np.zeros((self.n_epochs,2))

            #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
            self.motion_matrices = np.zeros((self.n_epochs,2,4))
            self.motion_matrices[:,0,0] = -1
            self.motion_matrices[:,1,1] = -1
            self.motion_matrices[:,0,2] = -self.obs_dyears
            self.motion_matrices[:,1,3] = -self.obs_dyears

            self.including_parallax = False
            self.n_fit_params = 4
        else:
            #calculate the parallax vector associated with the observation dates and (RA,Dec)
            self.parallax_vectors = delta_ra_dec_per_parallax_VECTORIZED(self.obs_times,self.gaia_time,ra,dec)

            #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
            self.motion_matrices = np.zeros((self.n_epochs,2,5))
            self.motion_matrices[:,0,0] = -1
            self.motion_matrices[:,1,1] = -1
            self.motion_matrices[:,0,2] = -self.obs_dyears
            self.motion_matrices[:,1,3] = -self.obs_dyears
            self.motion_matrices[:,0,4] = -self.parallax_vectors[:,0]
            self.motion_matrices[:,1,4] = -self.parallax_vectors[:,1]

            self.including_parallax = True
            self.n_fit_params = 5
            
        #calculate the final astrometry precision
        obs_roman_inv_covs = (self.roman_inv_covs[:,self.obs_roman_filt_inds]*self.obs_n_images_per_epoch[None,:,None,None])
        roman_summed_data_inv_covs = np.sum(np.einsum('oji,sojk->soik',
                                                      self.motion_matrices,
                                                      np.einsum('soij,ojk->soik',
                                                                obs_roman_inv_covs,
                                                                self.motion_matrices)),axis=1)

        #when only fitting for new positions
        final_pos_inv_covs = np.sum(obs_roman_inv_covs,axis=1)+self.gaia_inv_covs[:,:2,:2]        
        #when fitting positions and PMs, but no parallax
        final_pos_and_pm_inv_covs = self.gaia_inv_covs[:,:4,:4]+np.sum(np.einsum('oji,sojk->soik',
                                                                                  self.motion_matrices[:,:,:4],
                                                                                  np.einsum('soij,ojk->soik',
                                                                                            obs_roman_inv_covs,
                                                                                            self.motion_matrices[:,:,:4])),axis=1)
        final_astrometry_inv_covs = roman_summed_data_inv_covs+self.gaia_inv_covs[:,:self.n_fit_params,:self.n_fit_params]
        #for stability (like an extremely diffuse global prior on PM and parallax)
        final_astrometry_inv_covs[:,2:,2:] += np.diag(np.array([1e5,1e5,1e4])[:self.n_fit_params-2]**-2)

        self.final_astrometry_covs = np.zeros_like(final_astrometry_inv_covs)
        self.final_astrometry_covs[:] = np.nan
        self.final_astrometry_covs[(self.n_min_epochs_indv > 0)] = np.linalg.inv(final_astrometry_inv_covs[(self.n_min_epochs_indv > 0)])
        
        self.final_astrometry_precision = np.sqrt(self.final_astrometry_covs.T[np.arange(self.n_fit_params),
            np.arange(self.n_fit_params)])
        
        #use determinant definition of PM and Position error sizes
        self.final_pos_covs = self.final_astrometry_covs[:,:2,:2]
        self.final_pm_covs = self.final_astrometry_covs[:,2:4,2:4]
        if self.including_parallax:
            self.final_parallax_errs = np.sqrt(self.final_astrometry_covs[:,4,4])
        else:
            self.final_parallax_errs = self.gaia_parallax_errs
        self.final_pos_errs = np.power(np.linalg.det(self.final_pos_covs),1/4)
        self.final_pm_errs = np.power(np.linalg.det(self.final_pm_covs),1/4)
        
        return

    
                    
