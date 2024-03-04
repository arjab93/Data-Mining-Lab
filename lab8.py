# !pip install apriori
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv("store_data.csv", header=None)

records = []
for i in range(0, 7501):
    test = []
    data = dataset.iloc[i]
    data = data.dropna()
    for j in range(0, len(data)):
        test.append(str(dataset.values[i, j]))
    records.append(test)

association_rules = apriori(
    records, 
    min_support=0.005, 
    min_confidence=0.2, 
    min_lift=3, 
    min_length=2
)

association_results = list(association_rules)
for item in association_results:
    print(list(item.ordered_statistics[0].items_base), '->', list(item.ordered_statistics[0].items_add))
