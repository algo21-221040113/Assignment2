from market_profile import MarketProfile
import pandas as pd


amzn = pd.read_csv("IF_Data.csv").iloc[:20]
mp = MarketProfile(amzn)

mp_slice = mp[amzn.index.min():amzn.index.max()]