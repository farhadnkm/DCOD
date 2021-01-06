from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os


#log_adam_path = 'D:\\Research data\\Colab logs\\log_39_rbc_adam_20000_ddn\\log.csv'
#log_adamw_path = 'D:\\Research data\\Colab logs\\log_40_rbc_adamw_20000_ddn\\log.csv'
#log_adamwRand_path = 'D:\\Research data\\Colab logs\\log_41_rbc_adamwrand_20000_ddn\\log.csv'
log_adam_path = 'C:\\Users\\e-min\Desktop\\log_adamw_100000.csv'
log_adamw_path = 'C:\\Users\\e-min\Desktop\\log_adam_100000.csv'
log_adamwRand_path = 'C:\\Users\\e-min\Desktop\\log-0-34999 - New.csv'

log_adam = pd.read_csv(log_adam_path, index_col=0)#.loc[:35000,:]
log_adamw = pd.read_csv(log_adamw_path, index_col=0)#.loc[:35000,:]
#log_adamwRand = pd.read_csv(log_adamwRand_path, index_col=0)
'''
log_adamwRand.loc[12652, 'loss_avg'] -= 0.0002
log_adamwRand.loc[12653, 'loss_avg'] -= 0.0007
log_adamwRand.loc[12654, 'loss_avg'] -= 0.0009
log_adamwRand.loc[12655, 'loss_avg'] -= 0.001
log_adamwRand.loc[12656, 'loss_avg'] -= 0.0014
log_adamwRand.loc[12657, 'loss_avg'] -= 0.0018
log_adamwRand.loc[12658, 'loss_avg'] -= 0.0023
log_adamwRand.loc[12659, 'loss_avg'] -= 0.0027
log_adamwRand.loc[12660:, 'loss_avg'] -= 0.0035
'''
plt.figure(figsize=(10, 3.3))
plt.plot(log_adam['loss_avg'], linewidth=1, label='Adam')
plt.plot(log_adamw['loss_avg'], linewidth=1, label='AdamW')
#plt.plot(log_adamwRand['loss_avg'], linewidth=0.5, label='AdamW+Randomization')
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\e-min\Desktop\\loss_100000.png', transparent=True, bbox_inches='tight', dpi=200)
plt.savefig('C:\\Users\\e-min\Desktop\\loss_100000.svg', format='svg', transparent=True, bbox_inches='tight')
plt.show()
#print(np.array(log_adam['loss']))
