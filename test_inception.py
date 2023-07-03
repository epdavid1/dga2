from tsai.all import *
import pandas as pd
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

tst_noise = load_learner('inception_noise')
tst_test = pd.read_csv('test.csv')

iter = np.zeros(400)
for i in range(0,400):
	start = time.time()
	tst_noise.get_X_preds(tst_test.iloc[0].to_numpy().reshape(1,1,10001))	[2].astype(int)
	iter[i] = time.time()-start
print('Average detection time for 400 trials = '+str(int(np.round(np.mean(iter)*1000)))+' ms')
