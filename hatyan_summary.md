# Hatyan Python Subroutine Summary

Summary of all functions, classes, and methods defined in the Python files under `hatyan.git/hatyan/`.

---

## `__init__.py`

No functions or classes defined — serves as the package entry point, importing symbols from other modules.

---

## `hatyan_core.py`

Core routines for tidal constituent calculations.

- `check_requestedconsts(const_list_tuple, source)`: Validates that requested tidal components are available in the schureman or foreman source tables.
- `get_freqv0_generic(const_list, dood_date_mid, dood_date_start, source)`: Retrieves frequency and initial phase (v0) values for a list of tidal components.
- `get_uf_generic(const_list, dood_date_fu, nodalfactors, xfac, source)`: Gets nodal factors (u and f) for tidal components.
- `get_doodson_eqvals(dood_date, mode=None)`: Calculates Doodson astronomical values (T, S, H, P, N, P1) for the given dates.
- `robust_timedelta_sec(dood_date, refdate_dt=None)`: Generates timedelta in seconds, supporting dates outside the pandas DatetimeIndex range.
- `get_lunarSLSIHO_fromsolar(v0uf_base)`: Converts a Schureman v0uf table to different lunar conventions (SLS/IHO).
- `get_full_const_list_withfreqs()`: Retrieves the complete list of all available constituents with their frequencies.
- `sort_const_list(const_list)`: Sorts a constituent list by frequency.
- `get_const_list_hatyan(listtype)`: Returns predefined component lists by type (all, year, halfyear, month, day, etc.).

---

## `ddlpy_helpers.py`

Helpers for converting ddlpy (Dutch national data portal) measurement data to hatyan format.

- `ddlpy_to_hatyan(ddlpy_meas, ddlpy_meas_exttyp=None)`: Converts ddlpy measurement dataframes to hatyan timeseries dataframes, including unit conversion from cm to metres.
- `ddlpy_to_hatyan_plain(ddlpy_meas, isnumeric=True)`: Converts ddlpy plain measurements to hatyan format with values, qualitycode, and status columns.
- `convert_exttype_str2num(ts_measwl_ext, ts_measwl_exttype)`: Converts extreme type string codes (hoogwater, laagwater) to numeric HWLWcode values.

---

## `cli.py`

Command-line interface entry point.

- `cli(filename, overwrite, interactive_plots, redirect_stdout, loglevel)`: Initialises the output directory, runs hatyan configuration files, and manages logging for the CLI tool.

---

## `astrog.py`

Astronomical calculations for moon, sun, and tidal epochs.

- `astrog_culminations(tFirst, tLast, dT_fortran=False)`: Calculates lunar culminations, parallax, and declination over a timeframe.
- `astrog_phases(tFirst, tLast, dT_fortran=False)`: Calculates lunar phases (FQ, FM, LQ, NM) over a timeframe.
- `astrog_sunriseset(tFirst, tLast, dT_fortran=False, lon=5.3876, lat=52.1562)`: Calculates sunrise and sunset times at a given location.
- `astrog_moonriseset(tFirst, tLast, dT_fortran=False, lon=5.3876, lat=52.1562)`: Calculates moonrise and moonset times at a given location.
- `astrog_anomalies(tFirst, tLast, dT_fortran=False)`: Calculates lunar anomalies (perigee, apogee).
- `astrog_seasons(tFirst, tLast, dT_fortran=False)`: Calculates astronomical season boundaries.
- `astrab(date, dT_fortran=False, lon=5.3876, lat=52.1562)`: Calculates 18 astronomical parameters (moon/sun positions, altitudes, etc.) at a requested time.
- `astrac(timeEst, mode, dT_fortran=False, lon=5.3876, lat=52.1562)`: Refines astronomical time estimates for culminations, phases, etc.
- `get_leapsecondslist_fromurlorfile()`: Retrieves the leap seconds list from a URL or local file for time corrections.
- `dT(dateIn, dT_fortran=False)`: Calculates the difference between terrestrial time and universal time.
- `check_crop_dataframe(astrog_df, tFirst, tLast, tzone)`: Validates, crops, and applies a timezone to astronomical calculation results.
- `convert_str2datetime(tFirst, tLast)`: Converts string date inputs to datetime objects.
- `convert2perday(dataframeIn, timeformat='%H:%M %Z')`: Converts a datetime-indexed dataframe to per-day format.
- `plot_astrog_diff(pd_python, pd_fortran, ...)`: Plots differences between Python and Fortran astronomical calculation results.

---

## `utils.py`

General utility helpers.

- `close(fig=None)`: Wrapper around `matplotlib.pyplot.close()` for closing figures without a direct matplotlib import in calling code.

---

## `schureman.py`

Tidal constituent tables and nodal factors following the Schureman method.

- `get_schureman_shallowrelations()`: Retrieves Schureman shallow-water constituent relationships from the data file.
- `get_schureman_table()`: Calculates all Schureman constituents with v0uf values, including shallow-water components.
- `get_schureman_freqs(const_list, dood_date=..., return_allraw=False)`: Returns frequencies of requested tidal constituents.
- `get_schureman_v0(const_list, dood_date)`: Returns initial phase (v) values for constituents on specified date(s).
- `get_schureman_constants(dood_date)`: Returns fundamental astronomical constants used in nodal factor calculations.
- `get_schureman_u(const_list, dood_date)`: Returns u-values (phase corrections) for constituents.
- `get_schureman_f(const_list, dood_date, xfac)`: Returns f-values (amplitude factors) for constituents with optional x-factor corrections.
- `correct_fwith_xfac(f_i_pd, f_i_M2_pd, xfac)`: Applies x-factor corrections to f-values.

---

## `foreman.py`

Tidal constituent tables and nodal factors following the Foreman method.

- `get_foreman_doodson_nodal_harmonic(lat_deg=51.45)`: Retrieves Foreman harmonic constituents and nodal factors; result is latitude-dependent.
- `get_foreman_shallowrelations()`: Retrieves Foreman shallow-water constituent relationships with internal consistency checks.
- `get_foreman_doodson_nodal_all_NOTUSED(lat_deg=51.45)`: Combines Foreman harmonic and shallow constituents (not currently used).
- `get_foreman_v0_freq(const_list, dood_date=...)`: Retrieves v0 and frequency for Foreman constituents (harmonic and shallow water).
- `get_foreman_nodalfactors(const_list, dood_date)`: Calculates nodal factors (f and u) for Foreman constituents with satellite corrections.

---

## `timeseries.py`

Reading, writing, processing, and plotting of water-level timeseries.

- `calc_HWLW(ts, calc_HWLW345=False, buffer_hr=6)`: Calculates high and low water extremes using `scipy.signal.find_peaks` with an M2-period-based minimum distance.
- `calc_HWLWlocalto345(data_pd_HWLW, HWid_main)`: Converts local extremes to first LW (3), agger/double high water (4), and second LW (5) codes.
- `calc_HWLW12345to12(data_HWLW_12345)`: Converts 12345 HWLW codes to simplified 12 codes by finding the minimum water level after each HW.
- `filter_duplicate_hwlwnos(ts_ext)`: Identifies duplicate HWLW code+number combinations.
- `calc_HWLWnumbering(ts_ext, station=None)`: Assigns unique numbers to high/low waters based on tidal wave phase relative to the Cadzand reference point.
- `timeseries_fft(ts_residue, min_prominence, max_freqdiff, plot_fft, source)`: Performs FFT analysis on a residual timeseries and suggests matching tidal constituents.
- `plot_timeseries(ts, ts_validation=None, ts_ext=None, ts_ext_validation=None)`: Creates a comprehensive timeseries plot with optional validation data and extremes overlay.
- `plot_HWLW_validatestats(ts_ext, ts_ext_validation)`: Plots validation statistics for high/low water predictions.
- `write_netcdf(ts, filename, ts_ext=None, nosidx=False, mode='w')`: Writes a timeseries to netCDF file format, with optional extremes.
- `write_dia(ts, filename, headerformat='dia')`: Writes a timeseries to the HATYAN dia file format with a metadata header.
- `get_metadata_pd(ts, headerformat)`: Extracts and prepares metadata for timeseries file output.
- `dia_metadata_to_wia_metadata(metadata_pd, time_today, ana, grootheid)`: Converts dia metadata format to wia metadata format.
- `write_noos(ts, filename)`: Writes a timeseries to NOOS ASCII format.
- `crop_timeseries(ts, times, onlyfull=True)`: Crops a timeseries to a specified time range.
- `resample_timeseries(ts, timestep_min, tstart=None, tstop=None)`: Resamples a timeseries to a new timestep with interpolation.
- `nyquist_folding(ts_pd, t_const_freq_pd)`: Applies Nyquist frequency folding to prevent aliasing of tidal constituents.
- `check_rayleigh(ts_pd, t_const_freq_pd)`: Checks whether the Rayleigh criterion is met for constituent separation.
- `Timeseries_Statistics`: Class providing statistics for a timeseries (length, NaN count, timestep distribution).
- `get_diaxycoords(filename, crs)`: Extracts and reprojects coordinates from a dia file using pyproj.
- `get_diablocks_startstopstation(filename)`: Parses dia file metadata to identify data block locations and station names.
- `get_diablocks(filename)`: Comprehensive dia file parser that extracts all metadata blocks and parameters.
- `read_dia_nonequidistant(filename, diablocks_pd, block_id)`: Reads a non-equidistant (extreme) timeseries from a dia file.
- `read_dia_equidistant(filename, diablocks_pd, block_id)`: Reads an equidistant (regular interval) timeseries from a dia file.
- `read_dia(filename, station=None, block_ids=None, allow_duplicates=False)`: Main function to read dia files with station/block selection.
- `read_noos(filename, datetime_format='%Y%m%d%H%M', na_values=None)`: Reads timeseries files in NOOS ASCII format.

---

## `components.py`

Reading, writing, and plotting of tidal harmonic components.

- `plot_components(comp, comp_allperiods=None, comp_validation=None, sort_freqs=True)`: Creates an amplitude and phase plot for tidal components, with optional multi-period and validation overlays.
- `_get_tzone_minutes(tzone)`: Extracts the timezone offset in minutes from a timezone object.
- `write_components(comp, filename)`: Writes tidal components to a HATYAN format file with STAT/PERD/COMP headers.
- `merge_componentgroups(comp_main, comp_sec)`: Merges two component sets, with secondary components overwriting primary ones for matching constituents.
- `_read_components_analysis_settings(filename)`: Extracts analysis settings (xfac, nodalfactors) from component file comments.
- `_get_metadata_fromstarcomments(filename)`: Parses metadata from star-comment lines in component files.
- `_guess_xfactor_from_starcomments(filename)`: Infers whether the x-factor was applied from keywords in component file comments.
- `read_components(filename)`: Reads tidal components from a HATYAN format file, including metadata extraction.

---

## `metadata.py`

Metadata management for timeseries and component objects.

- `metadata_add_to_obj(obj, metadata)`: Adds a metadata dictionary to a pandas DataFrame's attrs without overwriting existing attrs.
- `metadata_from_diablocks(diablocks_pd, block_id)`: Extracts metadata from parsed dia file blocks.
- `metadata_from_ddlpy(ddlpy_meas)`: Extracts metadata from a ddlpy measurement dataframe.
- `metadata_from_obj(obj)`: Retrieves the metadata dictionary from a pandas DataFrame's attrs.
- `metadata_compare(metadata_list)`: Compares multiple metadata dictionaries and raises an error if they differ.
- `wns_from_metadata(metadata)`: Maps quantity/unit/vertref combinations to HATYAN waarnemingssoort (observation type) codes.

---

## `analysis_prediction.py`

Harmonic analysis and tidal prediction.

- `PydanticConfig`: Configuration class for pydantic compatibility.
- `MatrixConditionTooHigh`: Exception raised when the xTx matrix condition number exceeds its threshold.
- `HatyanSettings`: Settings class containing tidal analysis/prediction parameters (nodalfactors, fu_alltimes, xfac, source, etc.).
  - `__init__(...)`: Validates and stores all settings.
  - `__str__()`: Formats settings as a human-readable string.
- `vectoravg(A_all, phi_deg_all)`: Vector-averages constituent amplitudes and phases over multiple analysis periods.
- `analysis(ts, const_list, nodalfactors=True, fu_alltimes=True, xfac=False, source='schureman', ...)`: Performs harmonic tidal analysis on a timeseries to extract constituent amplitudes and phases.
- `analysis_singleperiod(ts, const_list, hatyan_settings)`: Performs harmonic analysis for a single period using least-squares fitting.
- `split_components(comp, dood_date_mid, hatyan_settings)`: Splits parent constituents into derived constituents using amplitude factors and phase increments.
- `prediction_singleperiod(comp, times, hatyan_settings)`: Generates a tidal prediction for specified times from a set of components.
- `prediction(comp, times=None, timestep=None)`: Main prediction function with optional timezone conversion and automatic time-range generation.

---

## `deprecated.py`

Backward-compatibility wrappers for renamed functions.

- `deprecated_python_option(**aliases)`: Decorator for deprecating function options with custom error messages.
- `check_old_kwargs(func_name, kwargs, aliases)`: Validates that deprecated function arguments are not passed.
- `get_components_from_ts(**kwargs)`: Deprecated alias for `analysis()`.
- `check_ts(**kwargs)`: Deprecated alias for `Timeseries_Statistics()`.
- `readts_dia(**kwargs)`: Deprecated alias for `read_dia()`.
- `readts_noos(**kwargs)`: Deprecated alias for `read_noos()`.
- `write_tsdia(**kwargs)`: Deprecated alias for `write_dia()`.
- `writets_noos(**kwargs)`: Deprecated alias for `write_noos()`.
- `write_tsnetcdf(**kwargs)`: Deprecated alias for `write_netcdf()`.
