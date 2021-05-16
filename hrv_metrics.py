#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from pyPreprocessing.smoothing import filtering
from pyRegression.nonlinear_regression import (nonlinear_regression,
                                               calc_function)


class hrv_metrics:
    """
    Class for metrics and calculations on interbeat interval data.

    Reference (open access):
    Shaffer F and Ginsberg JP (2017), An Overview of Heart Rate Variability
    Metrics and Norms. Front. Public Health 5:258.
    doi: 10.3389/fpubh.2017.00258

    The following time domain metrics are calculated:
        - mean: The average value of all IBIs
        - std: The standard deviation of all IBIs, also called SDNN or SDRR
        - min: The minimum value of all IBIs
        - max: The maximum value of all IBIs
        - p50: The percentage of adjacent IBIs that differ by more than 50 ms
        - RMSSD: The root mean square of successive IBI differences
        - HRV_TI: The HRV triangular index given as the ratio of the total
                  number of IBIs and the maximum abundance in the histogram.
                  Result depends on the binning size for histogram calculation.
        - TINN: The baseline width of the IBI histogram, determined by the
                width of a triangle function fitted to the histogram.

    The following time domain metrics are still missing:
        - SDANN: The standard deviation of IBIs from a longer measurement that
                 is divided into 5 min intervals.
        - SDNN Index: The standard deviation of SDANN values

    So far no frequency domain metrics are calculated.
    """

    def __init__(self, ibi_data):
        """
        Initialize the instance of hrv_metrics.

        Parameters
        ----------
        ibi_data : ndarray
            The interbeat intervals in chronological order in a one-dimensional
            array given in milliseconds. Missing values are allowed as np.nan.

        Returns
        -------
        None.

        """
        ibi_index = pd.Index(np.cumsum(ibi_data)/1000, name='time [s]')
        self.ibi_data = pd.DataFrame(ibi_data, columns=['ibi_raw'],
                                     index=ibi_index)
        self.ibi_data[('hr_raw')] = 60000/self.ibi_data['ibi_raw']

        self.ibi_data['ibi_smoothed'] = np.nan
        self.ibi_data['hr_smoothed'] = np.nan
        self.ibi_data['ibi_filtered'] = np.nan

        self.standard_preprocessing()
        self.descriptive_statistics()
        self.histogram_metrics()

    def standard_preprocessing(self):
        # Currently, no arguments are passed to self.filter_ibi() and
        # self.smooth_ibi. Should be added in the future to allow control over
        # filters applied.
        self.smooth_ibi('median', input_data='raw')
        self.filter_ibi('maximum', input_data='raw')
        self.filter_ibi('minimum', input_data='filtered')
        self.filter_ibi('lowpass', input_data='filtered')

    def smooth_ibi(self, mode, input_data='raw', **kwargs):
        """
        Apply smoothing to interbeat intervals (IBIs).

        Parameters
        ----------
        mode : str
            Allowed mode is 'median' (a rolling median filter is applied).
        input_data : str, optional
            Allowed values are 'raw' which defines that the input data for the
            smoothing is the raw data, or 'smoothed' which defined that the
            input data are smoothed data (only possible after initial run with
            'raw').
        **kwargs : TYPE
            median_window : int
                Is only needed if mode is 'median'. Must be an odd number.
                Default is 5.

        Returns
        -------
        None.

        """
        smoothing_data = self.ibi_data['ibi_{}'.format(input_data)].values

        smoothing_modes = ['median']

        if mode == smoothing_modes[0]:  # 'median'
            median_window = kwargs.get('median_window', 5)
            self.ibi_data['ibi_smoothed'] = median_filter(
                smoothing_data, size=median_window)
            self.ibi_data[('hr_smoothed')] = 60000/self.ibi_data[
                'ibi_smoothed']
        else:
            raise ValueError(
                'No valid mode entered. Allowed modes are {}'.format(
                    smoothing_modes))

    def filter_ibi(self, mode, input_data='raw', **kwargs):
        """
        Apply filters to interbeat intervals (IBIs).

        Filters are applied in the order maximum, minimum. The filtered
        data is stored in self.ibi_1d_processed. Filtered values are replaced
        by np.nan.

        Parameters
        ----------
        mode : str
            The working mode of the filter. Allowed values are 'maximum' (all
            values above a threshold are removed), 'minimum' (all values below
            a threshold are removed, 'lowpass' (data is filtered based on
            weighted moving average).
        input_data : str, optional
            Allowed values are 'raw' which defines that the input data for the
            filtering is the raw data, or 'filtered' which defined that the
            input data are filtered data (only possible after initial run with
            'raw').
        **kwargs for the different mode values:
            max_thresh : float
                Is only needed if mode=='maximum'. Default is 60000/25
                corresponding to a heart rate of 25 bpm.
            min_thresh : float
                Is only needed if mode=='minimum'. Default is 60000/220
                corresponding to a heart rate of 220 bpm.
            weights : list of float
                Only needed if mode=='lowpass'.
            std_factor : float
                Only needed if mode='lowpass'.

        Returns
        -------
        None.

        """
        filtering_data = self.ibi_data['ibi_{}'.format(input_data)].values

        filter_modes = ['maximum', 'minimum', 'lowpass']

        if mode == filter_modes[0]:  # maximum
            self.ibi_data['ibi_filtered'] = filtering_data
            maximum_threshold = kwargs.get('max_thresh', 60000/25)
            self.ibi_data['ibi_filtered'].iloc[
                self.ibi_data['ibi_filtered'] > maximum_threshold] = np.nan
        elif mode == filter_modes[1]:  # minimum
            self.ibi_data['ibi_filtered'] = filtering_data
            minimum_threshold = kwargs.get('min_thresh', 60000/220)
            self.ibi_data['ibi_filtered'].iloc[
                self.ibi_data['ibi_filtered'] < minimum_threshold] = np.nan
        elif mode == filter_modes[2]:  # lowpass
            weights = kwargs.get('weights', [1, 1, 0, 1, 1])
            std_factor = kwargs.get('std_factor', 2)
            self.ibi_data['ibi_filtered'] = np.squeeze(filtering(
                filtering_data[np.newaxis], 'spike_filter', weights=weights,
                std_factor=std_factor))
        else:
            raise ValueError(
                'No valid mode entered. Allowed modes are {}'.format(
                    filter_modes))

    def descriptive_statistics(self):
        self.time_domain_metrics = pd.DataFrame([],
                                                index=self.ibi_data.columns)
        self.time_domain_metrics['mean'] = np.nanmean(self.ibi_data.values,
                                                      axis=0)
        self.time_domain_metrics['std'] = np.nanstd(self.ibi_data.values,
                                                    axis=0)
        self.time_domain_metrics['min'] = np.nanmin(self.ibi_data.values,
                                                    axis=0)
        self.time_domain_metrics['max'] = np.nanmax(self.ibi_data.values,
                                                    axis=0)

        ibi_filtered_nonan = self.ibi_data['ibi_filtered'].values[
            ~np.isnan(self.ibi_data['ibi_filtered'].values)]
        # p50 is the percentage of successive IBIs that differ by more than
        # 50 ms
        self.time_domain_metrics.at['ibi_raw', 'p50'] = np.sum(
            np.diff(self.ibi_data['ibi_raw'].values) > 50)/len(
                self.ibi_data['ibi_raw'])
        self.time_domain_metrics.at['ibi_smoothed', 'p50'] = np.sum(
            np.diff(self.ibi_data['ibi_smoothed'].values) > 50)/len(
                self.ibi_data['ibi_smoothed'])
        self.time_domain_metrics.at['ibi_filtered', 'p50'] = np.sum(
            np.diff(ibi_filtered_nonan) > 50)/len(
                    self.ibi_data['ibi_filtered'])

        # RMSSD is the root mean square of successive IBI differences
        ibi_diffs = (self.ibi_data.values[:, 0:3] -
                     np.roll(self.ibi_data.values[:, 0:3], 1, axis=0))[1:]
        ibi_filtered_diffs = ibi_diffs[:, 2][~np.isnan(ibi_diffs[:, 2])]
        self.time_domain_metrics.loc[
            ['ibi_raw', 'ibi_smoothed'], 'RMSSD'] = np.sqrt(
                np.sum(ibi_diffs[:, 0:2]**2, axis=0)/ibi_diffs.shape[0])
        self.time_domain_metrics.at['ibi_filtered', 'RMSSD'] = np.sqrt(
            np.sum(ibi_filtered_diffs**2)/len(ibi_filtered_diffs))

    def histogram_metrics(self):
        # Geometric measures based on the IBI histograms
        # First generate histograms
        ibi_keys = ['ibi_raw', 'ibi_smoothed', 'ibi_filtered']
        hist_range = (self.time_domain_metrics.at['ibi_raw', 'min'],
                      self.time_domain_metrics.at['ibi_raw', 'max'])
        hist_range_diff = hist_range[1]-hist_range[0]
        bin_number = int(hist_range_diff/(1000/128))
        bin_size = hist_range_diff/bin_number
        self.ibi_histogram = pd.DataFrame([])
        for curr_key in ibi_keys:
            self.ibi_histogram[curr_key], bin_edges = np.histogram(
                self.ibi_data[curr_key], bins=bin_number, range=hist_range)
        bin_centers = (bin_edges + bin_size/2)[:-1]
        self.ibi_histogram.index = pd.Index(bin_centers,
                                            name='bin_center [ms]')
        # Find x and y values of histogram peaks
        hist_peak_values = self.ibi_histogram.max()
        hist_peak_ibis = self.ibi_histogram.idxmax()
        # HRV_TI is the HRV triangular index, the total number of IBI values
        # divided by the maximum in the histogram
        self.time_domain_metrics.loc[ibi_keys, 'HRV_TI'] = self.ibi_histogram[
            ibi_keys].sum()/hist_peak_values
        # Fit the histograms with a triangle function for TINN calculation
        self.histogram_triangle_fits = {}
        for curr_key in ibi_keys:
            self.histogram_triangle_fits[curr_key] = nonlinear_regression(
                bin_centers, self.ibi_histogram[curr_key].values, 'triangle',
                boundaries=[
                    (bin_centers[0], hist_peak_ibis[curr_key]),
                    (hist_peak_ibis[curr_key], bin_centers[-1]),
                    (hist_peak_ibis[curr_key], hist_peak_ibis[curr_key]),
                    (hist_peak_values[curr_key], hist_peak_values[curr_key])],
                alg='evo')
            self.ibi_histogram[curr_key + '_triangle'] = calc_function(
                bin_centers, self.histogram_triangle_fits[curr_key].x,
                'triangle')
            # TINN is the Triangular Interpolation of the NN Interval Histogram
            self.time_domain_metrics.at[curr_key, 'TINN'] = (
                self.histogram_triangle_fits[curr_key].x[1] -
                self.histogram_triangle_fits[curr_key].x[0])
