import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from scipy import io

# import data
n_inputs=10

# import data
file=io.loadmat('train_data.mat')
x=file['X_train']
y=file['y_train']
del file
#x_train=x_train[0:len(y_train),[1,2,3,4,5,6,7,8]] #no ws

x=x[0:len(y),[0,1,2,3,4,5,6,7,8,9]]
x=x.reshape(len(x),n_inputs)

# split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# set range
n_estimators_range = [250, 200, 150, 100,50]
min_samples_split_range = [3, 7, 11, 15, 19]

# save
results = []

# grid search
for n_est in n_estimators_range:
    for min_d in min_samples_split_range:
        model = RandomForestRegressor(n_estimators=n_est, min_samples_split=min_d)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results.append({'n_estimators': n_est, 'min_samples_split': min_d, 'R2': r2})

# heatmap
df = pd.DataFrame(results)
heatmap_data = df.pivot(columns='n_estimators', index='min_samples_split', values='R2')

# plot
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
            cbar_kws={'label': 'R-squared score'})
plt.title('Hyperparameter tuning heatmap', fontweight='bold')
plt.ylabel('Min samples split')
plt.xlabel('Number of estimators')
plt.tight_layout()
#plt.show()
plt.savefig("heatmap_train_min_samples_split.png", dpi = 300);