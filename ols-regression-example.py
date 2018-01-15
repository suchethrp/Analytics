# -*- coding: utf-8 -*-
##  removed cat1 and removed missing values fill

import pandas as pd
import statsmodels.formula.api as sm

## Read train data
train_df = pd.read_csv(
        '~/Downloads/train.csv',
        sep='\t',
        skipinitialspace=True,
        usecols=[
                'train_id',
                'name',
                'item_condition_id',
                'category_name',
    	        'brand_name',
                'price',
                'shipping',
                'item_description'
                ],
        nrows=20000,
        )
train_df.set_index('train_id', inplace=True)

## Fill blank values with unknown
train_df[['brand_name']] = train_df[['brand_name']].fillna(value='unknown')
train_df[['category_name']] = train_df[['category_name']].fillna(value='unknown/unknown/unknown')

## Split the category into seperate columns
foo = lambda x: pd.Series([i for i in x.split('/')])
cat_split = train_df['category_name'].apply(foo)
cat_split.rename(columns={0:'cat1',1:'cat2',2:'cat3'},inplace=True)
cat_split = cat_split[['cat1','cat2','cat3']]
train_df = train_df.join(cat_split)


## Build multi linear regression model
result = sm.ols(formula="price ~  shipping + item_condition_id + brand_name + cat1 + cat2 + cat3 ", data=train_df).fit()
print (result.summary())
