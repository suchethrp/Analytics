# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
import pandas as pd


from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def handle_na_data(data):
    data['category_name'].fillna(value='unknown/unknown/unknown', inplace=True)
    data['brand_name'].fillna(value='unknown', inplace=True)
    data['item_description'].fillna(value='unknown', inplace=True)

def get_most_common_dict(data, num):
    col_list=[]
    col_dict = Counter(data)

    for k, v in col_dict.most_common(num):
        col_list.append(k)

    return col_list


def handle_unknown_data(data, list_data):
    data = data.map(lambda s: 'unknown' if s not in list_data else s)

    return data

def to_categorical(data, col_list):
    def changeDataType(x):
         data[x] = data[x].astype('category')
    list(map(changeDataType, col_list))

    return data


def main():
    train = pd.read_table('~/Downloads/train.csv', engine='c', nrows=5000, sep='\t')
    test = pd.read_table('~/Downloads/test.csv', engine='c', nrows=500, sep='\t')
    submission = test[['test_id']]

    nrow_train = train.shape[0]

    merge = pd.concat([train, test])

    print('Finished Reading test data')

    handle_na_data(train)
    print('Finished handeling na data')

    #split category_name
    foo = lambda x: pd.Series([i for i in x.split('/')])
    cat_split = train['category_name'].apply(foo)
    cat_split.rename(columns={0:'cat1',1:'cat2',2:'cat3'},inplace=True)
    cat_split = cat_split[['cat1','cat2','cat3']]
    merge = merge.join(cat_split)

    merge.drop('category_name', axis=1, inplace=True)
    print('Finished splitling category_name data')

    brand_list = get_most_common_dict(merge['brand_name'], 30)
    cat1_list = get_most_common_dict(merge['cat1'], 10)
    cat2_list = get_most_common_dict(merge['cat2'], 20)
    cat3_list = get_most_common_dict(merge['cat3'], 300)

    print('Finished getting dict list data')

    merge['brand_name'] = handle_unknown_data(merge['brand_name'], brand_list)
    merge['cat1'] = handle_unknown_data(merge['cat1'], cat1_list)
    merge['cat2'] = handle_unknown_data(merge['cat2'], cat2_list)
    merge['cat3'] = handle_unknown_data(merge['cat3'], cat3_list)

    print('Finished handling unknown data')

    # merge = to_categorical(train, ['brand_name', 'cat1', 'cat2', 'cat3', 'item_condition_id'])
    x_dummies = csr_matrix(pd.get_dummies(merge[['brand_name','cat1','cat2','cat3']],sparse=True).values)
    print('Finished converting to dummy data')

    cv = CountVectorizer(min_df=10, stop_words='english')
    x_name = cv.fit_transform(merge['name'])
    print(type(x_name))
    print('Finished count vectorize `category_name`')

    tv = TfidfVectorizer(max_features=4000,
                         ngram_range=(1, 3),
                         stop_words='english')
    x_description = tv.fit_transform(merge['item_description'])
    print('Finished TFIDF vectorize `item_description`')

    sparse_merge = hstack((x_dummies, x_description, x_name)).tocsr()
    print('Finished to create sparse merge')

    X = sparse_merge[:nrow_train]
    y = np.log1p(train["price"])

    X_test = sparse_merge[nrow_train:]

    model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    print('Finished training model')

    submission = test[['test_id']]
    submission['price'] = np.expm1(model.predict(X=X_test))

    print(submission)

if __name__ == '__main__':
    main()
