
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


cat3_list =['Consoles',
    'Desktops & All-In-Ones',
    'Hobo',
    'Laptops & Netbooks',
    'Satchel',
    'Shoulder Bag',
    'Sweeping',
    'Totes&Shoppers',
    'unknown'
]

brands_list = ['Abbott',
'Acacia Swimwear',
'adidas Originals',
'Air Jordan',
'Alexander McQueen',
'Alexander Wang',
'Betty Boop',
'Bose',
'Bottega Veneta',
'Brahmin',
'Breville',
'Burberry',
'Canon',
'Cartier',
'Celine',
'Chanel',
'unknown'
]

def brandnew_flag(data):
    flag = 0
    if 'never used' in data.lower() or 'brand new' in data.lower():
        flag = 1
    return flag

def prom(data):
    flag = 0
    if 'prom' in data.lower():
        flag = 1
    return flag

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
        nrows=30000,
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


train_df['brand_name'] = train_df['brand_name'].map(lambda s: 'unknown' if s not in brands_list else s)
train_df['cat3'] = train_df['cat3'].map(lambda s: 'unknown' if s not in cat3_list else s)
train_df['new_flag'] = train_df['item_description'].map(brandnew_flag)
train_df['prom']= train_df['name'].map(prom)

##  get dummy values for categorical Data
train_df = pd.get_dummies(train_df, columns=['cat3'])
train_df = pd.get_dummies(train_df, columns=['brand_name'])

train_df.drop(['item_description','category_name','name', 'cat1', 'cat2'], axis=1, inplace=True)
y = train_df['price']
X = train_df.drop(['price'], axis=1)
clf = Ridge(alpha=1.0)
clf.fit(X, y)

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

y_predict = clf.predict(X_test)

print(clf.score(X_test, y_test))

print(y_predict)
