# gaia_roman_astrometry
Simulate combining Gaia catalogues with future Roman images to measure how much the resulting astrometry will be improved. The main notebook to work with is `gaia+roman_astrometry.ipynb`. The other pieces of code contain the main helper functions as well as measure the relationship between Roman position uncertainty and magnitude for different filters. 

Inside of `gaia+roman_astrometry.ipynb`, the user can set some key parameters like Roman observation times, magnitudes of sources in different filters, and the Gaia data-release era. At the end, summary figures show the size of the position, parallax, and proper motion uncertainties as well as compare those improvements to Gaia alone. Running the cells at the very bottom of the notebook will create plots showing the Gaia astrometric uncertainties as a function of G and Roman position uncertainties for different filters that are used for the astrometric calculations. 

### Caveats about Results:
* These calculations assume that there are enough stars in common between a Roman image and Gaia such that the uncertainty on the alignment contributes negligibly to Roman position uncertainty on the Gaia refrence frame. If a user would like to explore the impact of this uncertainty, they can increase the `roman_pos_err_floor` parameter to be larger than the nominal 1% of a pixel. 
* The parallax calculation places Roman at Earth instead of at L2 (because I already had the code for working with Earth), though the impact of this choice should be fairly minor on the astrometry precision calculations. This will need to be changed when working with real Roman data (or other telescopes at L2) and measuring the new astrometric vectors themselves.
* For numerical stability, there is an extremely diffuse global prior applied to the PM and parallax measurements (i.e. uncertainties are ~10 times larger than the largest possible stellar PM or parallax seen from Earth). This may lead to the output plots having Gaia+Roman PM uncertainties around 10^5 mas/yr and parallax uncertainties of 10^4 mas.

### Acknowledgements
* Scientifically inspired by the [Bayesian Positions, Parallaxes, Proper Motions (BP3M)](https://github.com/KevinMcK95/BayesianPMs) tool presented in [McKinnon et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...972..150M/abstract).
* Uses the predicted Gaia astrometric precision for positions, parallaxes, and proper motions for DR3 to DR5 as presented by the [Gaia Collaboration](https://www.cosmos.esa.int/web/gaia/science-performance).
* Uses the [`pandeia`](https://pypi.org/project/pandeia.engine/2025.5/#files) and [`stpsf`](https://stpsf.readthedocs.io/en/latest/index.html) packages to simulate Roman PSFs and observations.
* Thanks to [Eddie Schlafly](https://gist.github.com/schlafly) for sharing example code for calculating Roman astrometric uncertainties.
