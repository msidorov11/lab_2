import numpy as np

file_list = ['testx16x32_0.npz',    'testx16x32_2.npz',    'trainx16x32_0.npz',   'trainx16x32_2.npz',
'testx16x32_1.npz',    'testx16x32_3.npz',    'trainx16x32_1.npz',   'trainx16x32_3.npz',
'testx16x32_10.npz',   'testx16x32_4.npz',    'trainx16x32_10.npz',  'trainx16x32_4.npz',
'testx16x32_11.npz',   'testx16x32_5.npz',    'trainx16x32_11.npz',  'trainx16x32_5.npz',
'testx16x32_12.npz',   'testx16x32_6.npz',    'trainx16x32_12.npz',  'trainx16x32_6.npz',
'testx16x32_13.npz',   'testx16x32_7.npz',    'trainx16x32_13.npz',  'trainx16x32_7.npz',
'testx16x32_14.npz',   'testx16x32_8.npz',    'trainx16x32_14.npz',  'trainx16x32_8.npz',
'testx16x32_15.npz',   'testx16x32_9.npz',    'trainx16x32_15.npz',  'trainx16x32_9.npz']

data_all = [np.load('./ml-20mx16x32/' + i) for i in file_list]
merged_data = {}
for data in data_all:
    [merged_data.update({k: v}) for k, v in data.items()]
np.savez('ml-20mx16x32.npz', **merged_data)

file_npz = 'ml-20mx16x32.npz'
file_csv = 'ml-20mx16x32.csv'

file = np.load(file_npz)['arr_0']
np.savetxt(file_csv, file, fmt='%u')