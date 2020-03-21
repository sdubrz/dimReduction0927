import pandas as pd

main_path = 'D:\\Exp\\rawdata\\'
data_name = 'tic-tac-toe.csv'

data = pd.read_csv(main_path + data_name,header=None)
data = data.sample(n=250)
data.loc[:,0:8:].replace('x',-1).replace('b',0).replace('o',1).to_csv(main_path + 'data.csv', index=False,header=None)
data.loc[:,9].replace('positive',1).replace('negative',2).to_csv(main_path + 'label.csv', index=False,header=None)