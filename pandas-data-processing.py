# -*- coding: utf-8 -*-

## using pandas library processing/cleaning data
import pandas as pd

inputFilePath = '~/Downloads/train.csv'

input_df = pd.read_csv(
        inputFilePath,
        sep='\t',                                                  # delimiter
        usecols=['train_id', 'price', 'category_name'],            # columns name
        nrows=100,                                                 # number of rows
)

input_df.set_index('train_id', inplace=True)      # set index for DataFrame

# Empty vaues needs to be filled with some string or number for running and regression or classfication
# using pandas DataFrame fillna function to fill na values with a appropriate values
# filling category name with unknown/unknown/unknown beacuse , I will be splitting category names into 3 category

input_df[['category_name']] = input_df[['category_name']].fillna(value='unknown/unknown/unknown')

## Split the category into seperate columns
split_Cat = lambda x: pd.Series([i for i in x.split('/')]) # function to split category_name which is in form cat/cat/cat

cat_df = input_df['category_name'].apply(split_Cat)
cat_df.rename(columns={0:'cat1',1:'cat2',2:'cat3'},inplace=True) # rename the columns 0, 1, 2, to cat1, cat2, cat3
cat_df = cat_df[['cat1','cat2','cat3']]
input_df = input_df.join(cat_df)

# function to check if the price is > 10 , then set flag
def price_flag(price):
    flag = 0
    if price > 10:
        flag = 1
    return flag


input_df['price_flag'] = input_df['price'].map(price_flag)

print(input_df)
