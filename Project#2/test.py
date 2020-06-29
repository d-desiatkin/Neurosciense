import numpy as np
import pandas as pd

a = np.array([[1,2],[3,4]])
print(a)
print(np.mean(a, axis=0)/(np.mean(a, axis=0)*np.mean(a, axis=0)), np.mean(a, axis=0))

b = pd.DataFrame([[1,2],[3,4]]).iloc(1)
for i in b:
    print(np.array(i))