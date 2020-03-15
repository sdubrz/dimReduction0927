import pandas as pd

main_path = 'D:\\Exp\\rawdata\\'
data_name = 'abalone.csv'

data = pd.read_csv(main_path + data_name,header=None)
data = data.sample(n=250)
data.loc[:,1::].to_csv(main_path + 'data.csv', index=False,header=None)
data.loc[:,0].replace('M',1).replace('F',2).replace('I',3).to_csv(main_path + 'label.csv', index=False,header=None)