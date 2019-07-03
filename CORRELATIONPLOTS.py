import pandas as pd
import configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from fpdf import FPDF
from scipy import stats

dataNRC = pd.read_csv(cfg.NRC, header = None)
dataNRC = dataNRC.convert_objects(convert_numeric=True).as_matrix()
dataDND = pd.read_csv(cfg.DND, header = None)
dataDND = dataDND.convert_objects(convert_numeric=True).as_matrix()
xl_workbook = xlrd.open_workbook(cfg.headers)
x1_sheet = xl_workbook.sheet_by_index(0)
header = np.array([x1_sheet.cell_value(rowx=0, colx=cx) for cx in range(x1_sheet.ncols)])

def plotting2(x1,y1,filename,signal):
    plt.figure(figsize=(8,11))
    _,_,r,_,_ = stats.linregress(x1,y1)
    t = 'Correlation Plot for filename: '+filename+'\n signal: '+signal + '\n Correlation Coefficient: '+'{}'.format(r**2)
    plt.title(t)
    plt.plot(x1,y1,'.', mew=0,ms=2)
    plt.ylabel('DND Data')
    plt.xlabel('NRC Data')
    name = 'Correlation Plot for filename'+filename+' and '+(signal.replace(':', '')).replace('/','')+'.png'
    plt.savefig(name)
    #plt.show()
    plt.close()
    return name

rows=dataNRC.shape[0]
pdf = FPDF()
pdf.add_page()
placement = 1
for i in range(dataNRC.shape[1]):
    x = np.array(dataNRC[:rows,i])
    y = np.array(dataDND[:rows,i])
    n = plotting2(x,y,cfg.filename,'{}'.format(header[i]))
    if placement==1:
        pdf.image(n,5,8,h=125)
        placement = 2
    elif placement==2: 
        pdf.image(n,5,155, h=125)
        placement =  3
    elif placement ==3:
        pdf.image(n,100,8, h=125)
        placement=4
    else:
        placement=1
        pdf.image(n,100,155, h=125)
        pdf.add_page()
pdf.output("Correlation Plots for filename " + cfg.filename + ".pdf", "F")