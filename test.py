# import os
# filename = os.path.join('uesr','..','data')
# print(filename.split('/'))
# print(filename)
import pandas as pd
train_data = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
test_data = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
data = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
print(data)