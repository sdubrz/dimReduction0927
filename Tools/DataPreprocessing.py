import pandas as pd

main_path = 'D:\\Exp\\rawdata\\'
data_name = 'segment.csv'

data = pd.read_csv(main_path + data_name, header=None)

#data = data.loc[:, 1:10:]

data.dropna(how='any', axis=0, inplace=True)
data.drop_duplicates(keep='first',inplace=True)
data = data.sample(n=600)


print(data.duplicated().value_counts())

data.loc[:, 0:18:].to_csv(main_path + 'data.csv', index=False,header=None)
'''
lable = []
for i in range(0,data.shape[0]):
    lable.append([1])
df = pd.DataFrame(lable)
df.loc[:,:].to_csv(main_path + 'label.csv', index=False,header=None)
'''
data.loc[:, 19].to_csv(main_path + 'label.csv', index=False,header=None)