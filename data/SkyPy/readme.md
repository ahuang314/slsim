# Galaxy Simulation Configuration File
This YAML configuration file is used to generate simulated galaxy populations, including their redshift, luminosity, morphology, and other properties, using the SkyPy package. The file defines two populations of galaxies: blue galaxies (sources) and red galaxies (lenses), each with their unique parameters.

## Parameters
*  mag_lim: Magnitude limit of the galaxy sample (default: 35)
*  fsky: Sky area in square degrees (default: 0.1 deg^2) 
*  z_range: Redshift range for the galaxies
*  M_star_blue: Linear1D model for the blue galaxy population's characteristic absolute magnitude [Astropy-Linear1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Linear1D.html#)
*  phi_star_blue: Exponential1D model for the blue galaxy population's normalization [Astropy-Exponential1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Exponential1D.html)
*  alpha_blue: Faint-end slope of the blue galaxy population's Schechter luminosity function
*  M_star_red: Linear1D model for the red galaxy population's characteristic absolute magnitude [Astropy-Linear1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Linear1D.html#)
*  phi_star_red: Exponential1D model for the red galaxy population's normalization [Astropy-Exponential1D-github](https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Exponential1D.html)
*  alpha_red: Faint-end slope of the red galaxy population's Schechter luminosity function [Astropy-default_cosmology-github](https://docs.astropy.org/en/stable/api/astropy.cosmology.default_cosmology.html)
*  cosmology: Cosmological model used in the simulation

## Tables
The file contains two tables, one for each of the blue and red galaxies. Each table computes several properties for the galaxies, such as redshift, absolute magnitude, stellar mass, apparent magnitudes, physical size, angular size, and ellipticity. These properties are computed using various models and functions from the SkyPy package.

### 1. Blue Galaxy (Source Galaxies)
* z: Redshifts computed using the Schechter luminosity function [Skypy_schechter_lf_redshift](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.redshift.schechter_lf_redshift.html?highlight=skypy.galaxies.redshift.schechter_lf_redshift)
* M: Absolute magnitudes computed using the Schechter luminosity function [Skypy_schechter_lf_magnitude](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.luminosity.schechter_lf_magnitude.html?highlight=schechter_lf_magnitude)
* coeff: Dirichlet coefficients for the blue galaxy's spectrum (Spectral coefficients to calculate the rest-frame spectral energy) [Skypy_galaxies_spectrum-dirichlet_coefficients](https://skypy.readthedocs.io/en/stable/api/skypy.galaxies.spectrum.dirichlet_coefficients.html?highlight=galaxies.spectrum.dirichlet_coefficients)
* stellar_mass: Stellar masses computed using the kcorrect method
* mag_g, mag_r, mag_i, mag_z, mag_Y: Apparent magnitudes in LSST filters [Speclite_filters information](https://speclite.readthedocs.io/en/latest/filters.html?highlight=lsst%20filters#lsst-filters)
* physical_size: Physical sizes computed using the **late-type** lognormal size model [Skypy_late_type_lognormal_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.late_type_lognormal_size.html)
* angular_size: Angular sizes computed based on physical sizes and redshifts [Skypy_angular_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.angular_size.html)
* ellipticity: Ellipticities computed using the beta_ellipticity in skypy [Skypy_beta_ellipticity](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.beta_ellipticity.html)

### 2. Red Galaxy (Lensing Galaxies)
* z: Redshifts computed using the Schechter luminosity function
* M: Absolute magnitudes computed using the Schechter luminosity function
* coeff: Dirichlet coefficients for the red galaxy population's spectrum
* stellar_mass: Stellar masses computed using the kcorrect method
* mag_g, mag_r, mag_i, mag_z, mag_Y: Apparent magnitudes in LSST filters
* physical_size: Physical sizes computed using the **early-type** lognormal size model [Skypy_early_type_size](https://skypy.readthedocs.io/en/latest/api/skypy.galaxies.morphology.early_type_lognormal_size.html)
* angular_size: Angular sizes computed based on physical sizes and redshifts 
* ellipticity: Ellipticities computed using the beta_ellipticity model


# Dark Matter Halo Simulation Configuration File
This YAML configuration file is used to generate a simulated population of dark matter halos, including their redshifts and masses, using the colossus package and a custom pipeline.

### The halos pipeline require following packages:
* colossus
* hmf
* scipy

## Parameters
*  fsky: Sky area in square degrees (default: 0.0001 deg^2) 
* z_max: Maximum redshift for the halos (default: 5.00)
*  z_range: Redshift range for the halos !numpy.linspace 
*  cosmology: Cosmological model used in the simulation
*  m_min: Minimum halo mass in solar masses (default: 1.0E+12 M☉)
*  m_max: Maximum halo mass in solar masses (default: 1.0E+16 M☉)
* sigma8: Cosmological parameter (default: 0.81)
* ns: Cosmological parameter (default: 0.96)
* omega_m: Cosmological parameter (default: the same with the one in `cosmology`)

## Tables
The file constructs a table for dark matter halos, computing several key properties such as redshift and mass. 

### Dark Matter Halos
* **`z`**: Redshifts of dark matter halos, generated based on a given comoving density. The computation involves:
  - `redshift_list`: Array of redshifts ranging from 0 to `z_max=5.00`, divided into 100 intervals.
  - `sky_area`: The simulated sky area, set to `fsky=0.0001` square degrees.
  - `cosmology`: The cosmological model used, defaulting to Astropy's current default cosmology.
  - `m_min`: Minimum halo mass, set to `1.0E+12` solar masses (M☉).
  - `m_max`: Maximum halo mass, set to `1.0E+16` solar masses (M☉).
  - `resolution`: Set to 500 for halo redshift calculations.
  - `sigma8`, `ns`, `omega_m`: Cosmological parameters default set to 0.81, 0.96, and `omega_m` set same with the one in `cosmology` respectively.

* **`mass`**: Halo masses calculated at the generated redshifts, using:
  - `z`: The redshifts derived from the previous step.
  - Parameters `cosmology`, `m_min`, `m_max`, `resolution`, `sigma8`, `ns`, `omega_m` are utilized as defined above for `z`.

### Mass Sheet Correction
* **`z`**: Redshifts for mass sheet correction, calculated from the same comoving density as the halos but potentially with a different `resolution` of 200 for finer granularity in the correction process.
* **`kappa`**: Negative external convergence (kappa_ext) for each lensing sheet at the specified redshifts, computed using:
  - `redshift_list`: The redshifts calculated for mass sheet correction.
  - `first_moment`: The first moment of halo mass at these redshifts, determined through the `expected_mass_at_redshift` function with a `resolution` of 200.
  - Other parameters (`sky_area`, `cosmology`, `m_min`, `m_max`, `sigma8`, `ns`, `omega_m`) are consistent with those used for halo calculations.