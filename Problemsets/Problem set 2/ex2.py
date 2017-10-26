import sqlite3 as lite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

db = lite.connect('ThirteenDatasets.db')

for i in range(1,14):
	command = "select x,y from Set{0:02d}".format(i)
	#print(command)
	data = pd.read_sql_query(command, db)#.as_matrix()
	#print(data)
	print("\nMeans for Set{0:02d}:".format(i))
	print(data.mean(axis=0))
	print("\nStds for Set{0:02d}:".format(i))
	print(data.std(axis=0))
	
	sns.lmplot('x',
			   'y',
			   data = data,
			   fit_reg = True)
	plt.figure()		   
	sns.distplot(data['x'])
	sns.distplot(data['y'])
	plt.show()
	
