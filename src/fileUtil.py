import pandas as pd
from configure import Configure

class FileUtil:
	def csvToDataFrame(self, stock_name, normailzed = 0):
		url = '%s/%s' % (Configure.dataFolder, stock_name)
		col_names = ['Date','Open','High','Low','Close','Volume','Adj Close']
		stocks = pd.read_csv(url, header= 0, names= col_names)
		df = pd.DataFrame(stocks)
		date_split = df['Date'].str.split('-').str
		df['Year'], df['Month'], df['Day'] = date_split
		#take only four element	
		df.drop(df.columns[[0, 5,6,7,8,9]], axis= 1, inplace = True)
		
		if normailzed == 1:
			print("nothing is done")
		#dataframe.asmatrix()[0] = top of csv = latest data
		return df


