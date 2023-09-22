from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.galaxy_galaxy_lens import GalaxyGalaxyLens, theta_e_when_source_infinity
import numpy as np
from abc import ABC, abstractmethod
import warnings


class LensedSample(ABC):
    """
    Abstract Base Class to create a sample of lensed systems.
    All object that inherit from Lensed System must contain the methods it contains.
    """

    def __init__(self, sky_area=None, 
                 cosmo=None):
        """

        :param deflector_type: type of the lens
        :type deflector_type: string
        :param source_type: type of the source
        :type source_type: string
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self.cosmo = cosmo
        self.f_sky = sky_area
        if sky_area is None:
            from astropy.units import Quantity
            sky_area = Quantity(value=0.1, unit='deg2')
            warnings.warn("No sky area provided, instead uses 0.1 deg2")

        if cosmo is None:
            warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    @abstractmethod
    def generate_random_lensed_system(self):
        """
        draw a random lens within the cuts of the lens and source, with possible additional cut in the lensing
        configuration.

        # as well as option to draw all lenses within the cuts within the area

        :return: GalaxyGalaxyLens() instance with parameters of the deflector and lens and source light
        """
        pass
    
    @abstractmethod
    def potential_deflector_number(self):
        """
        number of potential deflectors (meaning all objects with mass that are being considered to have potential
        sources behind them)

        :return: number of potential deflectors
        """
        pass

    @abstractmethod
    def potential_source_number(self):
        """
        number of sources that are being considered to be placed in the sky area potentially aligned behind deflectors

        :return: number of sources
        """
        pass
    
    @abstractmethod
    def draw_sample(self):
        """
        return full sample list of all lenses within the area

        :return: List of LensedSystem instances with parameters of the deflectors and source.
        :rtype: list
        """

        pass 
