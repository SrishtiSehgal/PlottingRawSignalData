import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

#fixing random state for reproducibility
np.random.seed(19680801)

def norm(X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X = np.subtract(X, mean)
	print(std)
	X = np.divide(X, std)
	return X

#input
INPUT_PATH = input('filepath to signal files')+'\\'
filenames_used = [INPUT_PATH + i for i in os.listdir(INPUT_PATH) if '.csv' in i]
num_sig=int(input('number of signals'))
filename_names = input('vessel abbreviations separated by comma: ').split(',') 
maxi, mini = 0, float('inf')

#fill up data
for signal in range(num_sig):#number of signals
		data = []
		for i in (filenames_used):
			file = pd.read_csv(i).values[1:,signal]
			data.append((file)) #assumes one header row
			if (maxi < np.amax(file) ):
				maxi = np.amax(file)
			if (mini > np.amin(file) ):
				mini = np.amin(file)

		#graph details
		fig, ax1 = plt.subplots(figsize=(10, 6))
		fig.canvas.set_window_title('A Boxplot by Signal' +'{}'.format(signal))
		fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		star = dict(markerfacecolor='w', marker='*', markersize=6, markeredgecolor='black')
		bp = ax1.boxplot(data, showmeans=True, meanprops=star, patch_artist=True, notch=0, sym='+', vert=1, whis=1.5)
		plt.setp(bp['boxes'], color='black')
		plt.setp(bp['whiskers'], color='black')
		plt.setp(bp['fliers'], color='red', marker='+')
		
		# Add a horizontal grid to the plot, but make it very light in color
		# so we can use it for reading data values but not be distracting
		ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
		
		# Hide these grid behind plot objects
		ax1.set_axisbelow(True)
		ax1.set_title('Comparison of Data Across Five vessel for signal ' + '{}'.format(signal))
		ax1.set_xlabel('vessel name')
		ax1.set_ylabel('Value')

		# Set the axes ranges and axes labels
		ax1.set_xlim(0.5, len(filename_names) + 0.5)
		top = maxi + 50
		bottom = mini -50
		ax1.set_ylim(bottom, top)
		ax1.set_xticklabels(filename_names, rotation=45, fontsize=8)

		# Now fill the boxes with desired colors
		boxColors = ['darkkhaki', 'royalblue', 'green', 'violet', 'red', 'yellow', 'orange']
		medians = list(range(len(filename_names)))
		for i in range(len(filename_names)):
			bp['boxes'][i].set(facecolor=boxColors[i])

			# Now draw the median lines back over what we just filled in
			med = bp['medians'][i]
			medianX = []
			medianY = []
			for j in range(2):
				medianX.append(med.get_xdata()[j])
				medianY.append(med.get_ydata()[j])
				ax1.plot(medianX, medianY, 'k')
				medians[i] = medianY[0]
			
		# Due to the Y-axis scale being different across samples, it can be
		# hard to compare differences in medians across the samples. Add upper
		# X-axis tick labels with the sample medians to aid in comparison
		# (just use two decimal places of precision)
		pos = np.arange(len(filename_names)) + 1
		upperLabels = [str(np.round(s, 2)) for s in medians]
		weights = ['bold', 'semibold']
		for tick, label in zip(range(len(filename_names)), ax1.get_xticklabels()):
			ax1.text(pos[tick], top - (top*0.05), upperLabels[tick], horizontalalignment='center', size='x-small', weight=10, color=boxColors[tick])
		
		# Finally, add a basic legend
		#fig.text(0.80, 0.08, + ' Random Numbers', backgroundcolor=boxColors[0], color='black', weight='roman', size='x-small')
		
		f = plt.gcf()
		f.savefig(INPUT_PATH+'BoxPlot.png')
		plt.show()
