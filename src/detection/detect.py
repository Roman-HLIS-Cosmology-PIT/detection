import copy

from math import sqrt

import numpy as np
import numba as nb

import sep

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table

#from sf_tools.image.stamp import FetchStamps

from scipy.stats import median_abs_deviation as mad

import yaml

#import asdf


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


# def get_cutout(img, x, y, stamp_size):
#     fs = FetchStamps(img, int(stamp_size / 2))
#     x_round = np.round(x).astype(int)
#     y_round = np.round(y).astype(int)
#     dx = x_round - x
#     dy = y_round - y
#     fs.get_pixels(np.array([[y_round, x_round]]))
#     vign = fs.scan()[0].astype(np.float64)

#     return vign, dx, dy


@nb.njit(fastmath=True, cache=True)
def get_cutout_size(Qxx, Qxy, Qyy, n_sigma=3.0):
    # Compute trace and determinant
    trace = Qxx + Qyy

    # Compute eigenvalues analytically for symmetric 2x2 matrix
    temp = sqrt((Qxx - Qyy) ** 2 + 4 * Qxy**2)
    lam1 = 0.5 * (trace + temp)
    lam2 = 0.5 * (trace - temp)

    lam_max = max(lam1, lam2)

    return 2.0 * n_sigma * sqrt(lam_max)


def get_cutout(img, x, y, stamp_size):
    orow = int(y)
    ocol = int(x)
    half_box_size = stamp_size // 2
    maxrow, maxcol = img.shape

    ostart_row = orow - half_box_size + 1
    ostart_col = ocol - half_box_size + 1
    oend_row = orow + half_box_size + 2  # plus one for slices
    oend_col = ocol + half_box_size + 2

    ostart_row = max(0, ostart_row)
    ostart_col = max(0, ostart_col)
    oend_row = min(maxrow, oend_row)
    oend_col = min(maxcol, oend_col)

    cutout_row = y - ostart_row
    cutout_col = x - ostart_col

    return (
        img[ostart_row:oend_row, ostart_col:oend_col],
        cutout_row,
        cutout_col,
    )

def read_kernel(kernel_file_name):
    ## overwrite to DES kernel for now
    kernel = np.loadtxt(kernel_file_name)
    print ('kernel: ', kernel_file_name)
    #kernel = DES_KERNEL
    return kernel


def _normalize_photometry_config(phot_cfg):
    """
    Accept either:
      - photometry_method: 'kron'
      - photometry_method: {kron: {...}, aperture: {...}}
    Return dict of methods -> options dict.
    """
    if phot_cfg is None:
        return {}
    if isinstance(phot_cfg, str):
        return {phot_cfg: {}}
    if isinstance(phot_cfg, dict):
        # already a mapping of methods -> options
        return phot_cfg
    raise TypeError(f"Unsupported photometry_method config type: {type(phot_cfg)}")


def _add_photometry_columns(cat_dict, key_prefix, flux, fluxerr, flags, flux_radius):
    cat_dict[f"{key_prefix}_flux"] = flux
    cat_dict[f"{key_prefix}_fluxerr"] = fluxerr
    cat_dict[f"{key_prefix}_flags"] = flags
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(fluxerr > 0, flux / fluxerr, -10.0)
    cat_dict[f"{key_prefix}_snr"] = snr
    cat_dict[f"{key_prefix}_flux_radius"] = flux_radius
    return cat_dict

def read_asdf(filename):
    with asdf.open(filename) as af:
        data = af['data'] # what's going to be formatted like?
    return data

def read_fits(filename, sca):
    with fits.open(filename) as hdul:
        data = hdul[sca].data
        header = hdul[sca].header
    return data, header
def get_weights_from_image(img):
    weights = np.ones(img.shape)
    return weights
    

def get_cat(img_filename, config_file_name,sca = 1, header=None, wcs=None, mask=None):
    config = load_config(config_file_name)
    #Add a function to read image and weights from file
    img, header = read_fits(img_filename, sca)
    img = img.astype(float)
    weight = get_weights_from_image(img) # needs to be changed for better variable naming...
    #TODO!!!
    # NOTE: Might need to look again into this. For now we keep it simple.
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0
    origin = config.get("wcs_origin", 1)

    rms = np.median(np.sqrt(1 / weight[m]))
    # rms = mad(img, scale="normal", axis=(0, 1))

    if (header is not None) and (wcs is not None):
        raise ValueError("Only one of header or wcs can be provided.")
    elif header is not None:
        wcs = WCS(header)

    # NOTE: Sometimes we end up with a non-zero background, I don't know why..
    if config["background_subtraction"]:
        bkg = sep.Background(img, mask=mask_rms).globalback
    else:
        bkg = np.zeros_like(img) #will fix later

    img_sub = img - bkg

    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0

    rms = np.median(np.sqrt(1 / weight[m]))
    obj, seg = sep.extract(
        data=img_sub,
        thresh=config["detection_threshold"],
        err=rms,
        segmentation_map=config["segmentation_map"],
        minarea=config["min_area"],
        deblend_nthresh=config["deblend_nthresh"],
        deblend_cont=config["deblend_cont"],
        filter_type=config["filter_type"],
        filter_kernel=read_kernel(config["filter_kernel"]),
    )
    n_obj = len(obj)
    seg_id = np.arange(1, n_obj + 1, dtype=np.int32)

    # NOTE: keep old placeholders if other code relies on them, but prefer new per-method columns
    fluxes = np.ones(n_obj) * -10.0
    fluxerrs = np.ones(n_obj) * -10.0
    flux_rad = np.ones(n_obj) * -10.0
    snr = np.ones(n_obj) * -10.0
    flags = np.ones(n_obj, dtype=np.int64) * 64
    flags_rad = np.ones(n_obj, dtype=np.int64) * 64

    phot_cfg = _normalize_photometry_config(config.get("photometry_method"))

    # This dictionary will be merged into the output catalog/table later
    phot_cols = {}

    # ---- Kron photometry (per config) ----
    if "kron" in phot_cfg:
        kron_opts = phot_cfg.get("kron", {}) or {}
        kron_mult = float(kron_opts.get("multiplicative_factor", 2.5))
        phot_flux_frac = float(kron_opts.get("flux_rad_fraction", 0.5))

        kronrads, krflags = sep.kron_radius(
            data=img_sub,
            x=obj["x"],
            y=obj["y"],
            a=obj["a"],
            b=obj["b"],
            theta=obj["theta"],
            r=6.0,
            seg_id=seg_id,
            segmap=seg,
            mask=mask_rms,
        )

        good_kron = (
            (kronrads > 0)
            & (obj["b"] > 0)
            & (obj["a"] >= 0) #relaxing kron condition on a to allow circular sources
            & (obj["theta"] >= -np.pi/2)
            & (obj["theta"] <= np.pi/2)
        )
        kflux = np.ones(n_obj) * -10.0
        kfluxerr = np.ones(n_obj) * -10.0
        kflags = np.ones(n_obj, dtype=np.int64) * 64
        kflags[good_kron] = krflags[good_kron]
        kflux_rad = np.ones(n_obj) * -10.0
        kflags_rad = np.ones(n_obj, dtype=np.int64) * 64

        if np.any(good_kron):
            kflux_g, kfluxerr_g, kflag_g = sep.sum_ellipse(
                data=img_sub,
                x=obj["x"][good_kron],
                y=obj["y"][good_kron],
                a=obj["a"][good_kron],
                b=obj["b"][good_kron],
                theta=obj["theta"][good_kron],
                r=kron_mult * kronrads[good_kron],
                err = rms,
                subpix = 1,
                seg_id=seg_id[good_kron],
                segmap=seg,
                mask=mask_rms,
            )
            kflux[good_kron] = kflux_g
            kfluxerr[good_kron] = kfluxerr_g
            kflags[good_kron] |= kflag_g
        
            kflux_rad[good_kron], kflags_rad[good_kron] = sep.flux_radius(
                data=img_sub,
                x=obj["x"][good_kron],
                y=obj["y"][good_kron],
                rmax=6.0 * obj["a"][good_kron],
                frac=phot_flux_frac,
                normflux=kflux_g, #should be correct implementation of normflux
                subpix=5,
                seg_id=seg_id[good_kron],
                segmap=seg,
                mask=mask_rms,
            )    
        kflags_tot = kflags | kflags_rad | krflags
        phot_cols = _add_photometry_columns(phot_cols, "kron", kflux, kfluxerr, kflags_tot, kflux_rad)
        phot_cols["kron_radius"] = kronrads
        phot_cols["kron_multiplicative_factor"] = np.full(n_obj, kron_mult, dtype=float)

        #For objects where the radius is too small for the Kron radius, we replicate Source Extractor behavior as follows
        r_min = config.get("min_radius", 1.75)
        use_circle = kronrads * np.sqrt(obj["a"] * obj["b"]) < r_min    
        cflux, cfluxerr, cflag = sep.sum_circle(img_sub, obj["x"][use_circle], obj["y"][use_circle],
                                                r_min, subpix = 1)
        kflux[use_circle] = cflux
        kfluxerr[use_circle] = cfluxerr
        kflags_tot[use_circle] = cflag 

        # Optional: keep legacy single-method outputs if downstream expects them
        fluxes, fluxerrs, flags = kflux, kfluxerr, kflags_tot
        flux_rad = kflux_rad
        flags_rad = kflags_rad
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(fluxerrs > 0, fluxes / fluxerrs, -10.0)

    # ---- Aperture photometry (per config) ----
    if "aperture" in phot_cfg:
        ap_opts = phot_cfg.get("aperture", {}) or {}
        radii = ap_opts.get("radii", [])
        if not isinstance(radii, (list, tuple)) or len(radii) == 0:
            raise ValueError("photometry_method.aperture.radii must be a non-empty list")

        for r in radii:
            r = float(r)
            # sep.sum_circle returns (flux, fluxerr, flag)
            aflux, afluxerr, aflag = sep.sum_circle(
                data=img_sub,
                x=obj["x"],
                y=obj["y"],
                r=r,
                seg_id=seg_id,
                segmap=seg,
                mask=mask_rms,
            )
            # column-safe name (e.g., aper_r3p0)
            aflux_rad = np.ones(n_obj) * r
            rtag = str(r).replace(".", "p")
            key = f"aper_r{rtag}"
            phot_cols = _add_photometry_columns(phot_cols, key, aflux, afluxerr, aflag, aflux_rad)
            phot_cols[f"{key}_radius"] = np.full(n_obj, r, dtype=float)

    ra, dec = wcs.all_pix2world(obj["x"], obj["y"], origin)

    # Build the equivalent to IMAFLAGS_ISO
    # But you only know if the object is flagged or not, you don't get the flag
    ext_flags = np.zeros(n_obj, dtype=int)
    if mask is not None:
        for i, seg_id_tmp in enumerate(seg_id):
            seg_map_tmp = copy.deepcopy(seg)
            seg_map_tmp[seg_map_tmp != seg_id_tmp] = 0
            check_map = seg_map_tmp + mask
            if (check_map > seg_id_tmp).any():
                ext_flags[i] = 1

    out = Table()

    out["number"] = seg_id
    out["npix"] = obj["npix"]
    out["ra"] = ra
    out["dec"] = dec
    out["x"] = obj["x"]
    out["y"] = obj["y"]
    #out["x_mom"] = obj["x"]/np.sqrt(fluxes)
    #out["y_mom"] = obj["y"]/np.sqrt(fluxes)
    out["a"] = obj["a"]
    out["b"] = obj["b"]
    out["xx"] = obj["x2"]
    out["yy"] = obj["y2"]
    out["xy"] = obj["xy"]
    out["elongation"] = obj["a"] / obj["b"]
    out["ellipticity"] = 1.0 - obj["b"] / obj["a"]
    out["kronrad"] = kronrads
    out["flux"] = fluxes
    out["flux_err"] = fluxerrs
    out["flux_radius"] = flux_rad
    out["snr"] = snr
    out["flags"] = obj["flag"]
    out["flux_flags"] = krflags | flags | flags_rad
    out["ext_flags"] = ext_flags
    out["moment_radius"] = 0.5*np.sqrt(out["xx"]+out["yy"])

    # Merge photometry columns into the output catalog
    for k, v in phot_cols.items():
        out[k] = v

    return out, seg

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    out, seg = get_cat('', 'detect_config.yaml')
    out.write("test_catalog.fits", format = 'fits', overwrite = True)

    #plt.scatter(out['flux_radius'], 30-2.5*np.log10(out['flux']), alpha=0.3)
    #plt.show()


