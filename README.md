# gaia_roman_astrometry
Simulate combining Gaia catalogues with future Roman images to measure improved astrometry. The main notebook to be working with is `roman+gaia_astrometry.ipynb`. The other pieces of code contain the main helper functions as well as measure the relationship between Roman position uncertainty and magnitude for different filters. 

Inside of `roman+gaia_astrometry.ipynb`, the user can set some key parameters like Roman observation times, magnitudes of sources in different filters, and the Gaia data-release era. At the end, summary figures show the size of the position, parallax, and proper motion uncertainties as well as compare those improvements to Gaia alone.  

### Acknowledgements
* Scientifically inspired by the [Bayesian Positions, Parallaxes, Proper Motions (BP3M)](https://github.com/KevinMcK95/BayesianPMs) tool presented in [McKinnon et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJ...972..150M/abstract).
* Uses the predicted Gaia astrometric precision for positions, parallaxes, and proper motions for DR3 to DR5 as presented by the [Gaia Collaboration](https://www.cosmos.esa.int/web/gaia/science-performance).
* Uses the [`pandeia`](https://pypi.org/project/pandeia.engine/2025.5/#files) and [`stpsf`](https://stpsf.readthedocs.io/en/latest/index.html) packages to simulate Roman PSFs and observations.
* Thanks to [Eddie Schlafly](https://gist.github.com/schlafly) for sharing example code for calculating Roman astrometric uncertainties.

### Caveats:
* Assumes that there are enough stars in common between a Roman image and Gaia such that the uncertainty on the alignment contributes negligibly to Roman position uncertainty on the Gaia refrence frame.
* The parallax calculation places Roman at Earth instead of at L2, though the effect should be fairly minor on the astrometry precision calculations. This will need to be changed when working with real Roman data and measuring the new astrometric vectors themselves. 

