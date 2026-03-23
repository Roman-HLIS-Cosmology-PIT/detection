import copy
import sys

import astropy.units as u
import numpy as np
import sep
import yaml
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .config import SLIMFARMER_DEFAULT_CONFIG


def _sys_byteorder():
    return ">" if sys.byteorder == "big" else "<"


class ImageDetector:
    """
    Main class for detection method on various images.

    Parameters
    ----------
    bands : dict
        ``{band_name: {'science': path, 'psf': path, 'zeropoint': float,
                        'weight': path (optional)}}``
        All bands must be on the same pixel grid.
    detection_band : str, optional
        Band used for SEP source detection and model-type selection.
        Defaults to the first key in ``bands``.
    config_file : Path to config, optional
    """

    def __init__(self, bands, detection_band=None, config_file=None):
        self.config = SLIMFARMER_DEFAULT_CONFIG
        if config_file is not None:
            with open(config_file, "r") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.config.update(user_config)
        self.phot_cfg = self._normalize_photometry_config(self.config.get("photometry_method"))
        if detection_band is None:
            detection_band = list(bands.keys())[0]
        self.detection_band = detection_band
        self.bands = list(bands.keys())

        # ── Load all bands ────────────────────────────────────────────────────
        self.band_config = {}  # {band: {'science', 'weight', 'mask', 'zeropoint'}}
        self.wcs = None
        self.pixel_scale = None

        for band, bconf in bands.items():
            sci_path = bconf["science"]
            self.logger.info(f"Loading {band} from {sci_path}")
            with fits.open(sci_path) as hdul:
                ext = next((i for i, h in enumerate(hdul) if h.data is not None and h.data.ndim >= 2), 0)
                sci = hdul[ext].data.astype(np.float64)
                header = hdul[ext].header
            if self.wcs is None:
                self.wcs = WCS(header)
                pscl = proj_plane_pixel_scales(self.wcs) * u.deg
                self.pixel_scale = pscl[0].to(u.arcsec)

            wt_path = bconf.get("weight")
            if wt_path is not None:
                with fits.open(wt_path) as hdul:
                    ext = next((i for i, h in enumerate(hdul) if h.data is not None and h.data.ndim >= 2), 0)
                    wht = hdul[ext].data.astype(np.float64)
            else:
                _, _, rms = sigma_clipped_stats(sci[sci != 0])
                wht = np.where(rms > 0, np.ones_like(sci) / rms**2, 0.0)
                self.logger.info(f"  {band}: no weight — generated from clipped RMS = {rms:.4g}")

            bad = ~np.isfinite(sci)
            sci[bad] = 0.0
            wht[~np.isfinite(wht) | (wht < 0) | bad] = 0.0

            self.band_config[band] = {
                "science": sci,
                "weight": wht,
                "mask": (wht <= 0) | ~np.isfinite(sci),
                "zeropoint": bconf["zeropoint"],
            }

        # Shared base mask from detection band
        self.mask = self.band_config[detection_band]["mask"].copy()

    @staticmethod
    def _normalize_photometry_config(phot_cfg):
        if phot_cfg is None:
            return {}
        if isinstance(phot_cfg, str):
            return {phot_cfg: {}}
        if isinstance(phot_cfg, dict):
            return phot_cfg
        raise TypeError(f"Unsupported photometry config type: {type(phot_cfg)}")

    @staticmethod
    def _add_photometry_columns(cat_dict, key_prefix, flux, fluxerr, flags, flux_radius):
        cat_dict[f"{key_prefix}_flux"] = flux
        cat_dict[f"{key_prefix}_fluxerr"] = fluxerr
        cat_dict[f"{key_prefix}_flags"] = flags
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(fluxerr > 0, flux / fluxerr, -10.0)
        cat_dict[f"{key_prefix}_snr"] = snr
        cat_dict[f"{key_prefix}_flux_radius"] = flux_radius
        return cat_dict

    def _read_kernel(self):
        kernel_file = self.config.get("filter_kernel")
        print("kernel: ", kernel_file)
        return np.loadtxt(kernel_file)

    def _get_weights_from_image(self, img):
        if self.config.get("use_detection_weight"):
            wgt = self.band_config[self.detection_band]["weight"].copy()
            if wgt.dtype.byteorder not in ("=", "|", _sys_byteorder()):
                wgt = wgt.astype(wgt.dtype.newbyteorder("="))
            return wgt
        return np.ones(img.shape)

    def _read_fits(self, filename, sca):
        with fits.open(filename) as hdul:
            return hdul[sca].data, hdul[sca].header

    def _prepare_background(self, img):
        if self.config.get("background_subtraction", False):
            # Copying Chun-Hao's implementation here, not sure if this is always correct
            # in all cases
            return sep.Background(
                img,
                bw=self.config["back_bw"],
                bh=self.config["back_bh"],
                fw=self.config["back_fw"],
                fh=self.config["back_fh"],
            ).back()
        return np.zeros_like(img)

    def _run_kron_photometry(self, img_sub, obj, seg, seg_id, mask_rms, phot_cols):
        n_obj = len(obj)
        kron_opts = self.phot_cfg.get("kron", {}) or {}
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
            & (obj["a"] >= 0)
            & (obj["theta"] >= -np.pi / 2)
            & (obj["theta"] <= np.pi / 2)
        )

        kflux, kfluxerr = np.full(n_obj, -10.0), np.full(n_obj, -10.0)
        kflags, kflux_rad, kflags_rad = (
            np.full(n_obj, 64, dtype=np.int64),
            np.full(n_obj, -10.0),
            np.full(n_obj, 64, dtype=np.int64),
        )
        kflags[good_kron] = krflags[good_kron]

        if np.any(good_kron):
            kflux[good_kron], kfluxerr[good_kron], kflag_g = sep.sum_ellipse(
                data=img_sub,
                x=obj["x"][good_kron],
                y=obj["y"][good_kron],
                a=obj["a"][good_kron],
                b=obj["b"][good_kron],
                theta=obj["theta"][good_kron],
                r=kron_mult * kronrads[good_kron],
                err=self.rms_scalar,
                subpix=1,
                seg_id=seg_id[good_kron],
                segmap=seg,
                mask=mask_rms,
            )
            kflags[good_kron] |= kflag_g

            kflux_rad[good_kron], kflags_rad[good_kron] = sep.flux_radius(
                data=img_sub,
                x=obj["x"][good_kron],
                y=obj["y"][good_kron],
                rmax=6.0 * obj["a"][good_kron],
                frac=phot_flux_frac,
                normflux=kflux[good_kron],
                subpix=5,
                seg_id=seg_id[good_kron],
                segmap=seg,
                mask=mask_rms,
            )

        kflags_tot = kflags | kflags_rad | krflags
        phot_cols = self._add_photometry_columns(phot_cols, "kron", kflux, kfluxerr, kflags_tot, kflux_rad)
        phot_cols["kron_radius"] = kronrads

        r_min = self.config.get("min_radius", 1.75)
        use_circle = kronrads * np.sqrt(obj["a"] * obj["b"]) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(
            img_sub, obj["x"][use_circle], obj["y"][use_circle], r_min, subpix=1
        )

        kflux[use_circle], kfluxerr[use_circle], kflags_tot[use_circle] = cflux, cfluxerr, cflag

        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(kfluxerr > 0, kflux / kfluxerr, -10.0)

        return kflux, kfluxerr, kflags_tot, kflux_rad, kronrads, snr, phot_cols, krflags

    def _run_aperture_photometry(self, img_sub, obj, seg, seg_id, mask_rms, phot_cols):
        ap_opts = self.phot_cfg.get("aperture", {}) or {}
        radii = ap_opts.get("radii", [])
        if not isinstance(radii, list | tuple) or len(radii) == 0:
            raise ValueError("photometry_method.aperture.radii must be a non-empty list")

        for r in radii:
            r = float(r)
            aflux, afluxerr, aflag = sep.sum_circle(
                data=img_sub,
                x=obj["x"],
                y=obj["y"],
                r=r,
                seg_id=seg_id,
                segmap=seg,
                mask=mask_rms,
            )
            key = f"aper_r{str(r).replace('.', 'p')}"
            phot_cols = self._add_photometry_columns(
                phot_cols, key, aflux, afluxerr, aflag, np.full(len(obj), r)
            )
            phot_cols[f"{key}_radius"] = np.full(len(obj), r, dtype=float)
        return phot_cols

    def detect(self, img_filename, sca=1, header=None, wcs=None, mask=None):
        """
        Main detection function

        Parameters
        ----------
        img_filename : str
            Path to FITS file for detection (typically the science image of the detection band)
        sca : int, optional
            FITS extension index to read from img_filename, default is 1
        header : astropy.io.fits.Header, optional
            If provided, use this header for WCS instead of reading from img_filename
        wcs : astropy.wcs.WCS, optional
            If provided, use this WCS directly instead of reading from img_filename or header
        mask : 2D boolean array, optional
            Optional mask to flag pixels as bad (True = bad). If provided, will set ext_flags=1
            for any source with >0 masked pixels in its segmentation region.
        """
        if header is not None and wcs is not None:
            raise ValueError("Only one of header or wcs can be provided.")

        img, img_header = self._read_fits(img_filename, sca)
        img = img.astype(float)
        wcs = WCS(img_header) if header is None and wcs is None else (wcs or WCS(header))

        weight = self._get_weights_from_image(img)
        var = np.where(weight > 0, 1.0 / weight, 0.0)

        mask_rms = np.ones_like(weight)
        m = np.where(weight > 0)
        mask_rms[m] = 0

        bkg = self._prepare_background(img)
        img_sub = img - bkg

        sep.set_extract_pixstack(self.config.get("pixstack_size"))
        obj, seg = sep.extract(
            data=img_sub,
            thresh=self.config["detection_threshold"],
            var=var,
            segmentation_map=self.config["segmentation_map"],
            minarea=self.config["min_area"],
            deblend_nthresh=self.config["deblend_nthresh"],
            deblend_cont=self.config["deblend_cont"],
            filter_type=self.config["filter_type"],
            filter_kernel=self._read_kernel(),
        )

        n_obj = len(obj)
        seg_id = np.arange(1, n_obj + 1, dtype=np.int32)
        phot_cols = {}

        fluxes, fluxerrs = np.full(n_obj, -10.0), np.full(n_obj, -10.0)
        flux_rad, kronrads, snr = np.full(n_obj, -10.0), np.full(n_obj, -10.0), np.full(n_obj, -10.0)
        flags, krflags = np.full(n_obj, 64, dtype=np.int64), np.full(n_obj, 64, dtype=np.int64)

        if "kron" in self.phot_cfg:
            fluxes, fluxerrs, flags, flux_rad, kronrads, snr, phot_cols, krflags = self._run_kron_photometry(
                img_sub, obj, seg, seg_id, mask_rms, phot_cols
            )

        if "aperture" in self.phot_cfg:
            phot_cols = self._run_aperture_photometry(img_sub, obj, seg, seg_id, mask_rms, phot_cols)

        return self._build_catalog(
            obj, seg, seg_id, wcs, mask, fluxes, fluxerrs, flux_rad, kronrads, snr, flags, krflags, phot_cols
        )

    def _build_catalog(
        self,
        obj,
        seg,
        seg_id,
        wcs,
        mask,
        fluxes,
        fluxerrs,
        flux_rad,
        kronrads,
        snr,
        flags,
        krflags,
        phot_cols,
    ):
        n_obj = len(obj)
        ra, dec = wcs.all_pix2world(obj["x"], obj["y"], self.config.get("wcs_origin", 1))

        ext_flags = np.zeros(n_obj, dtype=int)
        if mask is not None:
            for i, sid in enumerate(seg_id):
                seg_map_tmp = copy.deepcopy(seg)
                seg_map_tmp[seg_map_tmp != sid] = 0
                if ((seg_map_tmp + mask) > sid).any():
                    ext_flags[i] = 1

        out = Table(
            {
                "number": seg_id,
                "npix": obj["npix"],
                "ra": ra,
                "dec": dec,
                "x": obj["x"],
                "y": obj["y"],
                "a": obj["a"],
                "b": obj["b"],
                "xx": obj["x2"],
                "yy": obj["y2"],
                "xy": obj["xy"],
                "elongation": obj["a"] / obj["b"],
                "ellipticity": 1.0 - obj["b"] / obj["a"],
                "kronrad": kronrads,
                "flux": fluxes,
                "flux_err": fluxerrs,
                "flux_radius": flux_rad,
                "snr": snr,
                "flags": obj["flag"],
                "flux_flags": krflags | flags,
                "ext_flags": ext_flags,
                "moment_radius": 0.5 * np.sqrt(obj["x2"] + obj["y2"]),
            }
        )

        for k, v in phot_cols.items():
            out[k] = v

        return out, seg


if __name__ == "__main__":
    detector = ImageDetector("detect_config.yaml")
    out, seg = detector.detect("")
    out.write("test_catalog.fits", format="fits", overwrite=True)
