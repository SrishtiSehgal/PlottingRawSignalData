import matplotlib.pyplot as plt
from matplotlib.dates import (DAILY, DateFormatter, MonthLocator, DayLocator,
                              rrulewrapper, RRuleLocator, drange)
import numpy as np
import datetime
import os 
import pandas as pd
# Fixing random state for reproducibility
np.random.seed(19680801)
loc = RRuleLocator(rrulewrapper(DAILY, byeaster=1))
formatter = DateFormatter('%m/%d/%y')
months = MonthLocator()
days = DayLocator()
all_filenames = [i for i in os.listdir() if '.csv' in i]
fig, ax = plt.subplots(len(all_filenames),1, sharex=True)
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for index,filename in enumerate(all_filenames):
	time, data, time_df = [], [], None
	print(filename)
	with open(filename, 'r') as input_file:
		for i, line in enumerate(input_file):
			if i > 0:
				split_line = line.split(',')
				time.append(split_line[0])
				data_point = split_line[2]
				if '.' in data_point:
					data.append(float(split_line[2]))
				else:
					data.append(int(split_line[2]))

	time_df = pd.to_datetime(pd.DataFrame(time).iloc[:,0])
	
	ax[index].plot_date(time_df, np.array(data).T, 
		markersize=0.1, color=colours[index])
	ax[-1].xaxis.set_major_locator(months)
	ax[-1].xaxis.set_major_formatter(DateFormatter('%m-%Y'))
	ax[-1].xaxis.set_minor_locator(days)
	ax[index].grid(True)
	ax[-1].set_xlabel("Datetime in mm-yy", fontsize=10)
	ax[index].set_ylabel("Signal", fontsize=10)
	ax[index].set_title(filename, fontsize=10)
	# rotates and right aligns the x labels, and moves the bottom of the
	# axes up to make room for them
	# fig.autofmt_xdate()

	ax[-1].xaxis.set_tick_params(rotation=30, labelsize=10)
fig1=plt.gcf()
plt.savefig('Signal_few_comp_raw.png')
plt.show()
