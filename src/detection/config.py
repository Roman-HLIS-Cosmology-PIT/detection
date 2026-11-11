"""Default configuration for detection module"""

import astropy.units as u

SLIMFARMER_DEFAULT_CONFIG = {
    # Detection (SEP)
    "thresh": 5.0,
    "minarea": 8,
    "back_bw": 32,
    "back_bh": 32,
    "back_fw": 2,
    "back_fh": 2,
    "filter_kernel": "gauss_2.0_5x5.conv",
    "filter_type": "matched",
    "deblend_nthresh": 2**8,
    "deblend_cont": 1e-10,
    "clean": True,
    "clean_param": 1.0,
    "pixstack_size": 1_000_000,
    "use_detection_weight": True,
    "photometry_method": "aperture",
    # Grouping
    "dilation_radius": 0.2 * u.arcsec,
    "group_buffer": 2.0 * u.arcsec,
    "group_size_limit": 10,
    "fit_dilation_radius": 0.2 * u.arcsec,  # expand fitting region beyond groupmap to capture profile wings
    # Background
    "subtract_background": False,
    "backtype": "variable",
    # Tractor optimisation
    "max_steps": 100,
    "damping": 1e-6,
    "dlnp_crit": 1e-3,
    "ignore_failures": True,
    "renorm_psf": 1.0,  # normalise PSF stamp to this value; 1=unbiased
}
