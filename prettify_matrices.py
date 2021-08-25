import pandas as pd
from glob import glob

paths = glob('csv/*.csv')
for path in paths:
    d = pd.read_csv(path, index_col=0)
    prefix = sorted(list(d.index))
    suffix = [ c for c in d.columns if c not in prefix ]
    d_new = pd.DataFrame(index=prefix, columns=prefix+suffix)
    for i in d.index:
        for j in d.columns:
            d_new.loc[i,j] = d.loc[i,j]
    d_new.fillna(0).to_csv(path.replace('csv/', 'processed/')[:-4] + '_pretty' + '.csv')
    