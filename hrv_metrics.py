#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, differential_evolution

from pyPreprocessing.smoothing import filtering


class hrv_metrics:
    """
    Class for metrics and calculations on interbeat interval data. Reference
    (open access):

    Shaffer F and Ginsberg JP (2017), An Overview of Heart Rate Variability
    Metrics and Norms. Front. Public Health 5:258.
    doi: 10.3389/fpubh.2017.00258
    """

    def __init__(self, ibi_data):
        """
        Initialize the instance of hrv_metrics.

        Parameters
        ----------
        ibi_data : ndarray
            The interbeat intervals in chronological order in a one-dimensional
            array. Missing values are allowed as np.nan.

        Returns
        -------
        None.

        """
        ibi_index = pd.Index(np.cumsum(ibi_data)/1000, name='time [s]')
        self.ibi_data = pd.DataFrame(ibi_data, columns=['ibi_raw'],
                                     index=ibi_index)
        self.ibi_data['ibi_smoothed'] = self.ibi_data['ibi_raw']
        self.ibi_data['ibi_filtered'] = self.ibi_data['ibi_raw']

        # Currently, no arguments are passed to self.filter_ibi() and
        # self.smooth_ibi. Should be added in the future to allow control over
        # filters applied.
        self.smooth_ibi('median')
        self.filter_ibi('maximum')
        self.filter_ibi('minimum')
        self.filter_ibi('lowpass')

        self.ibi_data[('hr_raw')] = 60000/self.ibi_data['ibi_raw']
        self.ibi_data[('hr_smoothed')] = 60000/self.ibi_data['ibi_smoothed']

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

        # Geometric measures based on the IBI histogram
        hist_range = (self.time_domain_metrics.at['ibi_raw', 'min'],
                      self.time_domain_metrics.at['ibi_raw', 'max'])
        hist_range_diff = hist_range[1]-hist_range[0]
        bin_number = int(hist_range_diff/(1000/128))
        bin_size = hist_range_diff/bin_number
        self.ibi_histogram = pd.DataFrame([])
        self.ibi_histogram['ibi_raw'], bin_edges = np.histogram(
            self.ibi_data['ibi_raw'], bins=bin_number, range=hist_range)
        self.ibi_histogram['ibi_smoothed'] = np.histogram(
            self.ibi_data['ibi_smoothed'], bins=bin_number,
            range=hist_range)[0]
        self.ibi_histogram['ibi_filtered'] = np.histogram(
            self.ibi_data['ibi_filtered'], bins=bin_number,
            range=hist_range)[0]
        self.ibi_histogram.index = pd.Index((bin_edges + bin_size/2)[:-1],
                                            name='bin_center [ms]')

        # self.fit_results = least_squares(
        #     triangle_residuals,
        #     [hist_range[0], hist_range[1],
        #      self.ibi_histogram['ibi_raw'].idxmax(),
        #      self.ibi_histogram['ibi_raw'].max()],
        #     bounds=([hist_range[0], hist_range[0], hist_range[0], 0],
        #             [hist_range[1], hist_range[1], hist_range[1], np.inf]),
        #     args=(self.ibi_histogram['ibi_raw'].values,
        #           self.ibi_histogram['ibi_raw'].index.values))

    def smooth_ibi(self, mode, **kwargs):
        """
        Apply smoothing to interbeat intervals (IBIs).

        Parameters
        ----------
        mode : str
            Allowed mode is 'median' (a rolling median filter is applied).
        **kwargs : TYPE
            median_window : int
                Is only needed if median is True. Must be an odd number.
                Default is 5.

        Returns
        -------
        None.

        """
        smoothing_modes = ['median']

        if mode == smoothing_modes[0]:  # 'median'
            median_window = kwargs.get('median_window', 5)
            self.ibi_data['ibi_smoothed'] = median_filter(
                self.ibi_data['ibi_smoothed'].values, size=median_window)
        else:
            raise ValueError(
                'No valid mode entered. Allowed modes are {}'.format(
                    smoothing_modes))

    def filter_ibi(self, mode, **kwargs):
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
            weighted moving average.
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
        filter_modes = ['maximum', 'minimum', 'lowpass']

        if mode == filter_modes[0]:  # maximum
            maximum_threshold = kwargs.get('max_thresh', 60000/25)
            self.ibi_data['ibi_filtered'].iloc[
                self.ibi_data['ibi_filtered'] > maximum_threshold] = np.nan
        elif mode == filter_modes[1]:  # minimum
            minimum_threshold = kwargs.get('min_thresh', 60000/220)
            self.ibi_data['ibi_filtered'].iloc[
                self.ibi_data['ibi_filtered'] < minimum_threshold] = np.nan
        elif mode == filter_modes[2]:  # lowpass
            weights = kwargs.get('weights', [1, 1, 0, 1, 1])
            std_factor = kwargs.get('std_factor', 2)
            self.ibi_data['ibi_filtered'] = np.squeeze(filtering(
                self.ibi_data['ibi_filtered'].values[np.newaxis],
                'spike_filter', weights=weights, std_factor=std_factor))
        else:
            raise ValueError(
                'No valid mode entered. Allowed modes are {}'.format(
                    filter_modes))

def calc_triangle(parameters, ibis):
    ibi_left = parameters[0]
    ibi_right = parameters[1]
    ibi_max = parameters[2]
    n_max = parameters[3]

    left_mask = np.logical_and(
        ibis >= min(ibi_left, ibi_max),
        ibis <= max(ibi_left, ibi_max))
    right_mask = np.logical_and(
        ibis > min(ibi_max, ibi_right),
        ibis <= max(ibi_max, ibi_right))

    left_slope = n_max/(ibi_max - ibi_left)
    right_slope = -n_max/(ibi_right - ibi_max)

    triangle = np.zeros_like(ibis)
    triangle[left_mask] = left_slope * (ibis[left_mask] - ibi_left)
    triangle[right_mask] = right_slope * (ibis[right_mask] - ibi_right)

    return triangle

def triangle_residuals(parameters, abundances, ibis):
    triangle = calc_triangle(parameters, ibis)

    residuals = np.sqrt(np.sum((abundances - triangle)**2))
    print(residuals)
    return residuals

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    y = np.array([0, 0, 0, 1, 2, 1, 0, 0, 0])
    x = np.arange(9)
    A = triangle_residuals([2,6,4,2], y, x)
    results=least_squares(triangle_residuals, [0,6,4,3], args=(y,x),
                          bounds=([0, 0, 0, 0], [9, 9, 9, 10]))
    # results = differential_evolution(triangle_residuals, [(0,4), (4,9), (0,9), (0,3)],
    #                                  args=(y,x))
    print(results.x)
    
    plt.scatter(x,y)
    plt.plot(x, calc_triangle(results.x, x))
    plt.show()