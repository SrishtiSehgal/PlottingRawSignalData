import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import config as cfg
from matplotlib.patches import Polygon
#fixing random state for reproducibility
np.random.seed(19680801)

def norm(X):
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X = np.subtract(X, mean)
	print(std)
	X = np.divide(X, std)
	return X

#fill up data
for signal in range(1):
    data = []
    for i in (cfg.filenames_used):
    	data.append((pd.read_csv(i).as_matrix()[1:,signal]))
    
    #graph details
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot by Signal' +'{}'.format(signal))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of Data Across Five filenames for signal ' + '{}'.format(signal))
    ax1.set_xlabel('filename')
    ax1.set_ylabel('Value')
    
    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue', 'green', 'violet', 'red']
    filename_names = ['CHA','STJ', 'MON-OCT', 'MON-AUG','VDQ']
    medians = list(range(5))
    for i in range(5):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = np.column_stack([boxX, boxY])
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[i])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='w', marker='*', markeredgecolor='k')
    
    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, 5 + 0.5)
    top = 600
    bottom = -100
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(filename_names,
                        rotation=45, fontsize=8)
    
    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(5) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(5), ax1.get_xticklabels()):
        ax1.text(pos[tick], top - (top*0.05), upperLabels[tick], horizontalalignment='center', size='x-small', weight=10, color=boxColors[tick])
    
    # Finally, add a basic legend
    #fig.text(0.80, 0.08, + ' Random Numbers', backgroundcolor=boxColors[0], color='black', weight='roman', size='x-small')
    
    plt.show()
