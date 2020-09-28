#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from suunto2python.suunto_exercise_data import exercise_data
from hrv_metrics.hrv_metrics import hrv_metrics


# Aug_27_2020 = exercise_data(
#     '/home/almami/Alexander/Suunto-Daten/entry_-335030377_1598542618/samples.json')
# Sep_5_2020 = exercise_data(
#     '/home/almami/Alexander/Suunto-Daten/entry_1353082632_1599316472/samples.json')
Sep_20_2020 = exercise_data(
    '/home/almami/Alexander/Suunto-Daten/entry_1993441714_1600629134/samples.json')
Sep_20_2020_ibi = hrv_metrics(Sep_20_2020.ibi_1d.values)
# Sep_21_2020_sleep = exercise_data(
#     '/home/almami/Alexander/Suunto-Daten/entry_1994106508_1600646290/samples.json')
Sep_22_2020_sleep = exercise_data(
    '/home/almami/Alexander/Suunto-Daten/entry_-814840220_1600732252/samples.json')
Sep_22_2020_ibi = hrv_metrics(Sep_22_2020_sleep.ibi_1d.values)

plot_ex_data = [Sep_22_2020_sleep]
plot_ibi_data = [Sep_20_2020_ibi, Sep_22_2020_ibi]

# # Plot of the gps coordinates passed during the exercise ('map').
# plt.figure(0)
# plt.scatter(
#     plot_data.exercise_data[('gps', 'Longitude')],
#     plot_data.exercise_data[('gps', 'Latitude')])

# # Plot of altitude, heart rate and pace over exercise time
# fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
# ax0.plot(plot_data.exercise_data.index[1:],
#          plot_data.exercise_data[('baro', 'Altitude')][1:])
# ax0.grid(True)
# ax0.set_xlabel('time')
# ax0.set_ylabel('altitude [m]')
# ax1.plot(plot_data.exercise_data.index[1:],
#          plot_data.exercise_data[('heart_rate', 'filtered')][1:])
# ax1.grid(True)
# ax1.set_xlabel('time')
# ax1.set_ylabel('heart rate [1/min]')
# ax2.plot(plot_data.exercise_data.index[1:],
#          plot_data.exercise_data[('gps', 'Pace')][1:])
# ax2.grid(True)
# ax2.set_xlabel('time')
# ax2.set_ylabel('pace [min/km]')
# ax2.set_ylim(4, 8)
# ax2.set_xlim(
#     plot_data.exercise_data.index[1], plot_data.exercise_data.index[-1])

fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax0.bar(Sep_20_2020_ibi.ibi_histogram['ibi_raw'].index.values,
        Sep_20_2020_ibi.ibi_histogram['ibi_raw'].values, width=7.8)
ax1.bar(Sep_20_2020_ibi.ibi_histogram['ibi_smoothed'].index.values,
        Sep_20_2020_ibi.ibi_histogram['ibi_smoothed'].values, width=7.8)
ax2.bar(Sep_20_2020_ibi.ibi_histogram['ibi_filtered'].index.values,
        Sep_20_2020_ibi.ibi_histogram['ibi_filtered'].values, width=7.8)

ax0.bar(Sep_22_2020_ibi.ibi_histogram['ibi_raw'].index.values,
        Sep_22_2020_ibi.ibi_histogram['ibi_raw'].values, width=7.8)
ax1.bar(Sep_22_2020_ibi.ibi_histogram['ibi_smoothed'].index.values,
        Sep_22_2020_ibi.ibi_histogram['ibi_smoothed'].values, width=7.8)
ax2.bar(Sep_22_2020_ibi.ibi_histogram['ibi_filtered'].index.values,
        Sep_22_2020_ibi.ibi_histogram['ibi_filtered'].values, width=7.8)

# fig_counter = 0
# for curr_data in plot_ibi_data:
#     # Plot of IBI values over time
#     all_ibis = curr_data.ibi_data['ibi_raw']
#     ibi_time_values = np.cumsum(all_ibis.values, axis=0)/1000/60# np.cumsum(np.diff(all_ibis.index.get_level_values(0)))
# #    ibi_time_values = np.concatenate(([pd.Timedelta(0)], ibi_time_values))
# #    ibi_time_values = pd.Series(ibi_time_values)/10**9

#     plt.figure(fig_counter)
#     ax3 = plt.subplot()
#     ax3.plot(ibi_time_values[:-1], all_ibis.values[:-1])
#     ax3.plot(ibi_time_values[:-1], curr_data.ibi_data['ibi_filtered'].values[:-1])
#     ax3.set_xlabel('time [min]')
#     ax3.set_ylabel('IBI [ms]')
#     ax3.set_ylim(300, 2000)
#     fig_counter += 1

#     # Poincaré-Plot of IBI values
#     plt.figure(fig_counter)
#     ax4 = plt.subplot()
#     ax4.scatter(all_ibis.values[:-2], np.roll(all_ibis.values[:-1], -1)[:-1])
#     ax4.scatter(curr_data.ibi_data['ibi_filtered'].values[:-2], np.roll(curr_data.ibi_data['ibi_filtered'].values[:-1], -1)[:-1])
#     ax4.set_xlabel('IBI(n) [ms]')
#     ax4.set_ylabel('IBI(n+1) [ms]')
#     ax4.set_xlim(300, 2000)
#     ax4.set_ylim(300, 2000)
#     fig_counter += 1
