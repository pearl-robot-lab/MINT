import pandas as pd
import os
import shutil
import glob
import math


for name in os.listdir('results'):
    idx=name.find('seed')
    if idx>-1:
        os.makedirs('results/'+name[:idx],exist_ok=True)
        shutil.move('results/'+name, 'results/'+name[:idx]+'/'+name)

for folder in list(glob.glob('results/*/')):
    files=os.listdir(folder)
    # if len(files)<5:
    #     continue
    results=[]
    for file in files:
        data=pd.read_excel(folder+file)
        results.append(data)
    results=pd.concat(results)
    mean=results.mean()
    std=results.std()
    confidence_interval=2*std / math.sqrt(5)
    data={}
    for key in mean.keys():
        data[key]=[mean[key],confidence_interval[key]]
    final_results=pd.DataFrame.from_dict(data, orient='index', columns=['mean','confidence_interval'])
    # final_results=final_results.style.format('{:.3f}')
    print()
    print(folder)
    print(final_results)