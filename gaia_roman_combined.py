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

#see here for WFI MultiAccum Table descriptions:
#https://roman-docs.stsci.edu/roman-instruments-home/wfi-imaging-mode-user-guide/observing-with-the-wfi-in-imaging-mode/wfi-multiaccum-tables
allowed_ma_names = ['IM_60_6_S','IM_66_6','IM_76_7_S','IM_85_7','IM_95_7','IM_101_7','IM_107_7','IM_107_8_S','IM_120_8',\
                    'IM_135_8','IM_152_9','IM_171_10','IM_193_11','IM_193_14_S','IM_225_13','IM_250_14','IM_284_14',\
                    'IM_294_16','IM_307_16','IM_360_16','IM_409_16','IM_420_16','IM_460_16','IM_500_16','IM_550_16',\
                    'IM_600_16','IM_650_16','IM_700_16','IM_750_16','IM_800_16','IM_900_16','IM_950_16','IM_1000_16']

#backgrounds and levels used when simulating centroiding accuracy
allowed_backgrounds = ['hltds','gbtds_mid_5stripe',\
                       'hlwas-medium_field1','hlwas-medium_field2',\
                       'hlwas-wide_field1','hlwas-wide_field2',\
                       'hlwas-wide_field3','hlwas-wide_field4']

# allowed_background_levels = ['high','medium','low']
allowed_background_levels = ['medium']

ma_integration_times = {}
for name in allowed_ma_names:
    ma_integration_times[name] = int(name.split('_')[1])

def roman_position_precision(mags_ab,filt,ma_name,
                             background,background_level,
                             datapath = datapath):
    '''
    For a given AB magnitude and Roman filter/MA/background,
    return the expected position uncertainty of a star.
    The underlying uncertainties come from running the 
    accompanying roman_astrometric_precision.ipynb notebook, which 
    uses pandeia and stpsf to estimate Roman position measurements.

    Inputs:
        mags_ab = numpy array of AB magnitudes to calculate position uncertainties
        filt = one of ['F062','F087','F106','F129',
                       'F146','F158','F184','F213'],  Roman filter
        ma_name = one of the choices in allowed_ma_names,  
                        Roman MultiAccum choice (i.e. exposure time)

    Outputs:
        sigma_xy = numpy array of position uncertainty of source in Roman image (mas)    
    '''

    if filt not in allowed_filters:
        print(f'Allowed filters are as follows:')
        print(allowed_filters)
        raise ValueError(f'ERROR: Chosen filter {filt} is not in the allowed filter list. Please try again.')
    if ma_name not in allowed_ma_names:
        print(f'Allowed MA names are as follows:')
        print(allowed_ma_names)
        raise ValueError(f'ERROR: Chosen MA name {ma_name} is not in the allowed MultiAccum list. Please try again.')
    if background not in allowed_backgrounds:
        print(f'Allowed backgrounds are as follows:')
        print(allowed_backgrounds)
        raise ValueError(f'ERROR: Chosen background of {background} is not in the allowed background list. Please try again.')
    if background_level not in allowed_background_levels:
        print(f'Allowed background levels are as follows:')
        print(allowed_background_levels)
        raise ValueError(f'ERROR: Chosen brackground level of {background_level} is not in the allowed background level list. Please try again.')

    background_string = f'{background}_{background_level}'
    fname = f'{datapath}roman_{ma_name}_{background_string}_{filt}_pos_errs.csv'
    if not os.path.isfile(fname):
        raise ValueError(f'ERROR: Could not find position error csv at {fname}'\
                         +f' Please download the appropriate data or run the roman_astrometric_precision.ipynb first.')
    input_data = pd.read_csv(fname)

    good_data_inds = np.where(np.isfinite(input_data['pos_errs_mas']))[0]
    return np.interp(mags_ab,input_data['mags_ab'][good_data_inds],input_data['pos_errs_mas'][good_data_inds],right=np.inf)


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

def create_motion_matrices(obs_times,ref_time,
                           ra=None,dec=None,
                           return_only_motion=True):
    '''
    takes 

    Inputs:
        obs_times = times of new observations, astropy list of Time objects
        ref_time = reference epoch time things are being measured from (e.g. gaia_time), single astropy Time object
        ra,dec = target/pointing coordinate in degrees, default is None, which ignores parallax in the following calculations
        return_only_motion = boolean, decides whether to just return motion_matrices or additional information

    Outputs:
        motion_matrices = motion_matrix associated with the observation dates and (RA,Dec)
        parallax_vectors = parallax vectors associated with the observation dates and (RA,Dec)
        including_parallax = boolean, True if parallax is used in fitting (i.e. real RA,Dec provided)
        n_fit_params = 4 or 5 depending on whether parallax is fit
    '''
    n_epochs = len(obs_times)
    obs_dyears = (obs_times-ref_time).to(u.year).value
    if (ra is None) or (dec is None):
        #then DO NOT include parallax in motion matrix calculations
        parallax_vectors = np.zeros((n_epochs,2))*np.nan

        #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
        motion_matrices = np.zeros((n_epochs,2,4))
        motion_matrices[:,0,0] = -1
        motion_matrices[:,1,1] = -1
        motion_matrices[:,0,2] = -obs_dyears
        motion_matrices[:,1,3] = -obs_dyears

        including_parallax = False
        n_fit_params = 4
    else:
        #then DO include parallax in motion matrix calculations

        #calculate the parallax vector associated with the observation dates and (RA,Dec)
        parallax_vectors = delta_ra_dec_per_parallax_VECTORIZED(obs_times,ref_time,ra,dec)

        #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
        motion_matrices = np.zeros((n_epochs,2,5))
        motion_matrices[:,0,0] = -1
        motion_matrices[:,1,1] = -1
        motion_matrices[:,0,2] = -obs_dyears
        motion_matrices[:,1,3] = -obs_dyears
        motion_matrices[:,0,4] = -parallax_vectors[:,0]
        motion_matrices[:,1,4] = -parallax_vectors[:,1]

        including_parallax = True
        n_fit_params = 5

    if return_only_motion:
        return motion_matrices
    else:
        return motion_matrices,parallax_vectors,including_parallax,n_fit_params

def delta_ra_dec_per_parallax_VECTORIZED(other_times,gaia_time,ra,dec):
    '''
    Calculate the parallax offset vectors at other_times for a position (ra,dec)

    Inputs:
        other_times = times you want parallax vectors at, list of astropy.Time objects
        gaia_time = reference epoch, astropy.Time object
        ra = RA (degrees) of target/pointing
        dec = Dec (degrees) of target/pointing

    Outputs:
        parallax_vectors = numpy array giving parallax offsets in RA,Dec for each time in other_times,
                            parallax_offset = parallax_vectors * parallax, where parallax is in mas
    '''
    
    #THIS CODE IS REPURPOSED FROM CODE FROM MELODIE KAO
    
    #choose any parallax because we will scale by it later
    parallax = 1.0*u.mas
    distance = (1/parallax.value)*u.kpc
    delta_time = (gaia_time-other_times).to(u.year).value
    dates = other_times[:,None]+(np.array([delta_time*0,delta_time]).T)*u.year
    dates = Time(dates, format='mjd')

    #need sun position at the observation times
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

    #calculate offsets in alpha*,delta (i.e. change in RA,Dec)
    #for a 1 mas parallax
    dalpha = cos_alpha*cos_ecc*sin_lam-sin_alpha*cos_lam
    ddelta = cos_delta*sin_ecc*sin_lam-cos_alpha*sin_delta*cos_lam\
                -sin_alpha*sin_delta*cos_ecc*sin_lam
    parallax_vectors = np.zeros((len(other_times),2))
    parallax_vectors[:,0] = dalpha
    parallax_vectors[:,1] = ddelta
    
    return parallax_vectors


class gaia_roman_astrometric_precision:
    """
    takes lists of Roman magnitudes (and corresponding filter names) and Gaia magnitudes
    to be used in determining the astrometric improvement from combining telescopes

    provide ra,dec coordinates (in degrees) to properly account for parallax in the 
    calculations, otherwise parallax will be ignored and only positions and PM 
    precisions will be updated.

    Inputs:
        roman_mags = numpy array, roman magnitudes in different filters of all stars, shape = (n_stars,n_filters)
        roman_filters = list of filters used, shape = (n_filters)
        gaia_mags = numpy array, gaia magnitudes, shape = (n_stars)
        observation_list = description of observations, shape = (n_observations,(epoch_MJD,roman_filter,N_images_at_epoch))
        roman_pos_floor_err = uncertainty floor (in mas) of Roman position measurements, default is 1% of pixel width
        gaia_ref_epoch = Gaia reference epoch, default is J2016.0
        ra,dec = target/pointing coordinate in degrees, default is None, which ignores parallax in the following calculations
        roman_background = simulated background used for Roman position uncertainty, default is hlwas-wide_field1
        roman_background_level = simulated background level used for Roman position uncertainty, default is medium
    """
    
    def __init__(self, roman_mags, roman_filters, gaia_mags, observation_list,
                 gaia_era = 'DR4', roman_pos_floor_err = 0.01*roman_pix_scale, 
                 gaia_ref_epoch = 2016.0, ra = None, dec = None, 
                 roman_background = 'hlwas-wide_field1', 
                 roman_background_level = 'medium'):

        """
        for all the given roman magnitudes, save the corresponding position uncertainties (mas),
        as well as Gaia-based astrometric precision covariance matrices given the Gaia mags

        then use the observation information to update the astrometry uncertainties
        """

        #general information
        self.n_stars = len(gaia_mags)
        self.n_epochs = len(observation_list)
        self.ra = ra
        self.dec = dec

        #roman information
        self.n_filters = len(roman_filters)
        self.roman_filters = np.array(roman_filters)
        self.roman_mags = roman_mags
        self.roman_background = roman_background
        self.roman_background_level = roman_background_level
        self.roman_pos_floor_err = roman_pos_floor_err

        #save unique combinations of MA/exposure time and Roman filters
        self.obs_unique_filter_MAs = np.unique(np.array(observation_list)[:,[1,3]],axis=0)
        self.obs_unique_filter_MAs_strings = np.zeros(len(self.obs_unique_filter_MAs)).astype(str)
        for ind,pair in enumerate(self.obs_unique_filter_MAs):
            self.obs_unique_filter_MAs_strings[ind] = '_'.join(pair)
        self.obs_n_unique_filt_MAs = len(self.obs_unique_filter_MAs)

        #define Roman position uncertainties for relevant MA/filter combinations
        self.roman_covs = np.zeros((self.n_stars,self.obs_n_unique_filt_MAs,2,2))
        self.roman_pos_errs = np.zeros((self.n_stars,self.obs_n_unique_filt_MAs))
        for filt_ind in range(self.obs_n_unique_filt_MAs):
            filt = self.obs_unique_filter_MAs[filt_ind][0]
            ma_name = self.obs_unique_filter_MAs[filt_ind][1]
            
            self.roman_pos_errs[:,filt_ind] = roman_position_precision(self.roman_mags[:,np.where(self.roman_filters == filt)[0][0]],
                                                                       filt,ma_name,
                                                                       roman_background,
                                                                       roman_background_level)
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
        self.obs_roman_ma_names = np.zeros(self.n_epochs).astype(str)
        for obs_ind in range(self.n_epochs):
            self.obs_epoch_mjds[obs_ind] = observation_list[obs_ind][0]
            self.obs_n_images_per_epoch[obs_ind] = observation_list[obs_ind][2]
            self.obs_roman_ma_names[obs_ind] = observation_list[obs_ind][3]
            
            filt = observation_list[obs_ind][1]
            ma_name = observation_list[obs_ind][3]
            self.obs_roman_filt_inds[obs_ind] = np.where(self.obs_unique_filter_MAs_strings == '_'.join([filt,ma_name]))[0][0]

        self.n_min_epochs_indv = np.sum(self.good_roman_errs[:,self.obs_roman_filt_inds],axis=1)
        self.n_min_epochs_indv[self.good_gaia_pos] += 1
        self.n_min_epochs_indv[self.good_gaia_pms] += 1
        self.multi_epoch_stars = (self.n_min_epochs_indv > 1)
        
        self.obs_times = Time(self.obs_epoch_mjds, format='mjd')

        self.motion_matrices,\
                    self.parallax_vectors,\
                    self.including_parallax,\
                    self.n_fit_params = create_motion_matrices(self.obs_times,
                                                               self.gaia_time,
                                                               ra=self.ra,dec=self.dec,
                                                               return_only_motion=False)
        
        # self.obs_dyears = (self.obs_times-self.gaia_time).to(u.year).value
        # if (ra is None) or (dec is None):
        #     #then DO NOT include parallax in motion matrix calculations
        #     self.parallax_vectors = np.zeros((self.n_epochs,2))

        #     #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
        #     self.motion_matrices = np.zeros((self.n_epochs,2,4))
        #     self.motion_matrices[:,0,0] = -1
        #     self.motion_matrices[:,1,1] = -1
        #     self.motion_matrices[:,0,2] = -self.obs_dyears
        #     self.motion_matrices[:,1,3] = -self.obs_dyears

        #     self.including_parallax = False
        #     self.n_fit_params = 4
        # else:
        #     #then DO include parallax in motion matrix calculations

        #     #calculate the parallax vector associated with the observation dates and (RA,Dec)
        #     self.parallax_vectors = delta_ra_dec_per_parallax_VECTORIZED(self.obs_times,self.gaia_time,ra,dec)

        #     #motion_matrix dot [dracosdec,ddec,pmracosdec,pmdec,] = position change
        #     self.motion_matrices = np.zeros((self.n_epochs,2,5))
        #     self.motion_matrices[:,0,0] = -1
        #     self.motion_matrices[:,1,1] = -1
        #     self.motion_matrices[:,0,2] = -self.obs_dyears
        #     self.motion_matrices[:,1,3] = -self.obs_dyears
        #     self.motion_matrices[:,0,4] = -self.parallax_vectors[:,0]
        #     self.motion_matrices[:,1,4] = -self.parallax_vectors[:,1]

        #     self.including_parallax = True
        #     self.n_fit_params = 5
            
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

        self.roman_alone_astrometry_inv_covs = roman_summed_data_inv_covs
        self.final_astrometry_inv_covs = final_astrometry_inv_covs
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

    
                    
