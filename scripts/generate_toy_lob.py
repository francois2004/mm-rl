import numpy as np
import pandas as pd

np.random.seed(42)

N = 200
dt = 1/254

mu = 0. ##
sigma = 0.02
mid = [100]

for i in range (N-1):
    dS = mu*mid[-1]*dt + sigma*mid[-1]*np.sqrt(dt)*np.random.randn()
    mid.append(mid[-1]+dS)

mid = np.array(mid)

spread = .01
bid = mid-spread/2
ask = mid+spread/2

bid_vol = np.random.poisson(10,N)
ask_vol = np.random.poisson(10,N)

df = pd.DataFrame({ "mid" : mid, "bid" : bid, "ask": ask, "bid_vol" : bid_vol, "ask_vol" :ask_vol})
df.to_csv("data/raw/toy_lob.csv", index = False)
