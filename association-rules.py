# script reads transactions from file and optputs assocation rules
# support is set to 1% to capture more data
# python assocation-rules.py 'input-file.csv'

import sys
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

transactions = pd.read_csv(sys.argv[1],
                            sep=',',
                            usecols=['transaction_id', 'item_id'],
                         )


basket_sets = transactions.groupby(['transaction_id', 'item_id'])['item_id'].count().unstack().reset_index().fillna(0).set_index('transaction_id')

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules.head(50))
