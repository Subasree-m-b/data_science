import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

#Load Data
df1 = pd.read_csv("chennai_house_price.csv")
pd.set_option("display.max_columns",None)
pd.set_option("display.width",1000)
# print(df1.head())
# print(df1.shape)
# print(df1.columns)

#histogram on BHK
# plt.hist(df1.bhk,rwidth=0.8)
# plt.xlabel("No. of BHK")
# plt.ylabel("Count")
# plt.show()

#removing unnecessary columns
df2 = df1.drop(['status','builder'],axis='columns')
# print(df2.head())
# print(df2.isnull().sum())

#handling NA values
df3=df2.dropna(subset=["age"])
# print(df3.isnull().sum())
# print(df3.shape)

#filling NAN values
df3["bathroom"]=df3['bathroom'].fillna(df3['bhk'] - 1)
# print(df3.isnull().sum())
# print(df3.shape)

#add new feature called price per square feet
df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['area']
# print(df4.head())

#dimensionality reduction for location
df4["location"] = df4["location"].apply(lambda x: x.strip())
location_stats = df4['location'].value_counts(ascending=False)
# print(location_stats)
# print(location_stats[location_stats>10])
# print(len(location_stats[location_stats<=10]))

"""Any location having less than 10 data points should be tagged as "other" location. 
This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, 
it will help us with having fewer dummy columns"""

location_stats_less_than_10 = location_stats[location_stats<=10]
# print(location_stats_less_than_10)
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(len(df4.location.unique()))
# print(df4.head(10))
# print(df4.shape)

"""As a data scientist when you have a conversation with your business manager 
(who has expertise in real estate), he will tell you that normally square ft per 
bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 
400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. 
We will remove such outliers by keeping our minimum threshold per bhk to be 300 sqft"""

# print(df4[df4.area/df4.bhk<300].head())
# print(len(df4[df4.area/df4.bhk<300]))

"""These are clear data errors that can be removed safely"""
df5= df4[~(df4.area/df4.bhk<300)]
# print(df5.head())
# print(df5.shape)

# print(df5.price_per_sqft.describe())

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (10, 20)
    plt.scatter(bhk2.area, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.area, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()

# plot_scatter_chart(df5, "Madhavaram")

"""We should also remove properties where for same location, the price of (for example) 3 bedroom
 apartment is less than 2 bedroom apartment (with same square ft area).Now we can remove those 2 BHK
apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment"""
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df6 = remove_bhk_outliers(df5)
df7 = df6.copy()
# print(df7.shape)

"""Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties"""
# plot_scatter_chart(df7, "Porur")

#remove unwanted columns
# print(df7.head())
df8=df7.drop(["price_per_sqft"],axis="columns")
# print(df8.head())

#Use One Hot Encoding For Location
dummies = pd.get_dummies(df8.location)
# print(dummies.head(3))
df9 = pd.concat([df8,dummies.drop('other',axis='columns')],axis='columns')
# print(df9.head())
df10 = df9.drop('location',axis='columns')
# print(df10.head(2))

#Building a model
# print(df10.shape)
X = df10.drop(['price'],axis='columns')
# print(X.head(3))
y = df10.price
# print(y.head(3))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
# print(lr_clf.score(X_test,y_test))

#Use K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(), X, y, cv=cv))

"""We can see that in 5 iterations we get a score above 80% all the time. 
This is pretty good but we want to test few other algorithms for regression 
to see if we can get even better score. We will use GridSearchCV for this purpose"""

#Find best model using GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept':[True, False],
                'positive':[True,False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['squared_error','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

result = find_best_model_using_gridsearchcv(X,y)
# print(result)

"""Based on above results we can say that LinearRegression gives the best score. Hence we will use that."""
#Test the model for few properties

def predict_price(location,area,bathroom,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = area
    x[1] = bathroom
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# print(predict_price('Selaiyur',1000, 2, 2))
print(predict_price('Porur',1000, 3, 3))
print(predict_price('Pammal',1000, 2, 2))
