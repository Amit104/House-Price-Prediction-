import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score 
import pandas as pd
import numpy as np
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.tree import DecisionTreeRegressor
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)    
    from sklearn.cross_validation import train_test_split
    from sklearn import cross_validation
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA	
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats
import pickle


features_to_input = ['bedrooms','bathrooms','sqft_living',
                    'sqft_lot','floors','waterfront','view','condition',
                    'grade','sqft_above','yr_built','yr_renovated',
                    'zipcode','lat','long','sqft_living15','sqft_lot15']

models = [LinearRegression(), GradientBoostingRegressor(), RandomForestRegressor(), 
                DecisionTreeRegressor()]

model_names = ["Linear Regression", "Gradient Boosting Regression", "Random Forest Regression", 
                "Decision Tree Regression"]
   

'''
Preprocess data to remove noise, duplicates and outliers.

@return : processed data
'''
def preprocess_data(full_dataset):
    print('Handling nan data...')
    full_dataset.fillna(full_dataset.mean(),inplace=True)
    dupes=full_dataset.duplicated()
    
    print('Handling dulicates...')
    print('-->',sum(dupes), ' duplicate values found and removed')

    print('Handling outliers...')
    full_dataset[(np.abs(stats.zscore(full_dataset)) < 3).all(axis=1)]
    return full_dataset

'''
feature scaling of data

@param X : features
@return  : scaled features
'''
def feature_normalize(X):
    mean1 = []
    std1 = []
    X_norm = X
    c = X.shape[1]
    for i in range(c):
        m = np.mean(X[:,i])
        s = np.std(X[:,i])
        mean1.append(m)
        std1.append(s)
        X_norm[:,i] = (X_norm[:,i]-m)/s
 
    return X_norm,mean1,std1

'''
feature scaling of data with given mean and standard deviation

@param X : features, mean, std
@return  : scaled features
'''
def feature_norm_given(X,m,s):
    X_norm = X
    c = X.shape[1]
    for i in range(c):
        X_norm[:,i] = (X_norm[:,i]-m[i])/s[i]
    return X_norm


'''
Load data using pandas.

@return: pandas dataframe containing dataset data.
'''
def load_data():
    print('Loading Dataset...')
    full_dataset = pd.read_csv('price.csv')
    full_dataset = full_dataset[['price','bedrooms','bathrooms','sqft_living',
                                'sqft_lot','floors','waterfront','view','condition',
                                'grade','sqft_above','yr_built','yr_renovated',
                                'zipcode','lat','long','sqft_living15','sqft_lot15']]
    return full_dataset


'''
Get and validate user input.

@return: input value 
'''
def user_input():
    user = input()
    while(True):
        try:
            int(user)
        except ValueError:
            print('Enter integer value')
            user = input()
        else:
            return user

'''
To allow user to input custom features row to get predictions of price.

@param df: house features dataset 
@return: feature array of user input.
'''
def user_feature(df):
    
    print('Do you want to enter input? 1 - Yes, 2-No')
    user = int(user_input())

    if(user==2):
        print('Thank you!!!')
        return None
    else:
        print("Enter Feature Values for the following, enter -1 if want to leave blank")
        X_pred = []
        for i in features_to_input:
            print(i)
            f = float(user_input())

            if(f==-1):
                X_pred.append(df[i].mean())
            else:
                X_pred.append(f)
    temp = []
    temp.append(X_pred)
    X_pred = np.array(temp)
    print(X_pred)
    return X_pred


'''
Plots a bar chart comparing accuracies of various classifiers used in the cascaded classifier.

@param TestModels : Contains Model name and Accuracy
'''
def plot_comparison(TestModels):
    y = []
    xT = []

    for i in TestModels.Model:
        xT.append(i)

    for i in TestModels.R2_Price:
        y.append(i)

    x = range(0,len(xT))
    plt.xticks(x, xT)
    plt.xticks(range(len(x)), xT, rotation=45)
    plt.bar(x,y)
    plt.title('Comparisons of classifiers')
    print('')
    input('Press any key for the plot to appear...')
    plt.show()



'''
Train and test the dataset on cascaded classifiers.
Used methods such as PCA.
Individual classifiers in first layer include Linear Regression, Gradient Boosting Regression, 
Random Forest Regression, Decision Tree Regression.
Second layer has Gradient Boosting Regression with dataset formed using output from first 
layer classifiers result.

@param X : features
@param y : labels
@return  : object containing accuracy of each classifier.
'''
def check_accuracy(X,y):
    print('Train test split...')
    Xtrn, Xtest, Ytrn, Ytest = cross_validation.train_test_split(X,y,test_size=0.4)

    TestModels = pd.DataFrame()
    tmp = {}

    
    print('\n Individual classifiers results...')

    upd_features_blending_pca = []
    upd_features_blending = []
    upd_features_stacking = []
    ct = -1
    indices = []
    for model in models:
        ct += 1
        model.fit(Xtrn, Ytrn)   
        res = model.predict(Xtest)
        new = []
        new.append(res)
        res = np.array(new)
        if(upd_features_blending_pca==[]):
            pca = PCA(n_components=11)
            pca.fit(X)
            X_pca = pca.transform(Xtest)
            identity17 = np.identity(17)
            coeff = pca.transform(identity17)
            #pickle the coeff
            filename = 'saved/coeff'
            pickle.dump(coeff, open(filename, 'wb'))
            upd_features_blending_pca = X_pca
            upd_features_blending = Xtest
            upd_features_stacking = np.array(res).T
        else:
            upd_features_stacking = np.insert(upd_features_stacking,[len(upd_features_stacking[0])],res.T,axis=1)

        upd_features_blending = np.insert(upd_features_blending,[len(upd_features_blending[0])],res.T,axis=1)
        upd_features_blending_pca = np.insert(upd_features_blending_pca,[len(upd_features_blending_pca[0])],res.T,axis=1)

        print('[', ct + 1 ,'] Accuracy using', model_names[ct],'...')
        print(model.score(Xtest,Ytest))
        tmp['Model'] = model_names[ct]
        tmp['R2_Price'] = model.score(Xtest,Ytest)
        TestModels = TestModels.append([tmp])
        #pickle the model
        filename = 'saved/' + model_names[ct]
        pickle.dump(model, open(filename, 'wb'))

    print('\n Cascaded classifiers result...')
    ct=0
    ct+=1
    print('[',ct,'] Accuracy with PCA and blending, using Gradient Boosting regression...')

    Xtrn_pca_blending, Xtest_pca_blending, Ytrn_pca_blending, Ytest_pca_blending = cross_validation.train_test_split(upd_features_blending_pca,Ytest,test_size=0.2)
    model1 = GradientBoostingRegressor().fit(Xtrn_pca_blending, Ytrn_pca_blending)
    print(model1.score(Xtest_pca_blending,Ytest_pca_blending))
    tmp['Model'] = 'PCA + blending'
    tmp['R2_Price'] = model1.score(Xtest_pca_blending,Ytest_pca_blending)
    TestModels = TestModels.append([tmp])
    # pickle the model
    filename = 'saved/pca_blend'
    pickle.dump(model1, open(filename, 'wb'))
    
    ct+=1
    print('[',ct,'] Accuracy with blending, using Gradient Boosting regression...')

    Xtrn_blending, Xtest_blending, Ytrn_blending, Ytest_blending = cross_validation.train_test_split(upd_features_blending,Ytest,test_size=0.2)
    model2 = GradientBoostingRegressor().fit(Xtrn_blending, Ytrn_blending)
    print(model2.score(Xtest_blending,Ytest_blending))
    tmp['Model'] = 'blending'
    tmp['R2_Price'] = model2.score(Xtest_blending,Ytest_blending)
    TestModels = TestModels.append([tmp])
    # pickle the model
    filename = 'saved/blend'
    pickle.dump(model2, open(filename, 'wb'))
    
    ct+=1
    print('[',ct,'] Accuracy with stacking, using Gradient Boosting regression...')
    Xtrn_stacking, Xtest_stacking, Ytrn_stacking, Ytest_stacking = cross_validation.train_test_split(upd_features_stacking,Ytest,test_size=0.2)
    model3 = GradientBoostingRegressor().fit(Xtrn_stacking, Ytrn_stacking)
    print(model3.score(Xtest_stacking,Ytest_stacking))
    tmp['Model'] = 'stacking'
    tmp['R2_Price'] = model3.score(Xtest_stacking,Ytest_stacking)
    TestModels = TestModels.append([tmp])
    #pickle the model
    filename = 'saved/stacking'
    pickle.dump(model3, open(filename, 'wb'))
    
    return TestModels


'''
user the above trained models to predict user input price

@param X : user input feature set.
@return  : predictions.
'''
def predict_user_input(X):
    models = {}
    ct = 1
    for i in model_names:
        models[ct] = pickle.load(open("saved/"+i,"rb" ))
        ct = ct + 1
    print('\n Loading Price predictions...')

    model1 = (pickle.load(open("saved/pca_blend",'rb')))
    model2 = (pickle.load(open("saved/blend",'rb')))
    model3 = (pickle.load(open("saved/stacking",'rb')))
    coeff = (pickle.load(open("saved/coeff",'rb')))

    upd_features_blending_pca = []
    upd_features_blending = []
    upd_features_stacking = []
    i = 0
    pred_vals = []
    for ct in range(1,5):
        print('\n',model_names[i],'-->')
        i = i + 1
        print(models[ct].predict(X))
        res = models[ct].predict(X)
        pred_vals.append(res[0])
        new = []
        new.append(res)
        res = np.array(new)
        if(upd_features_stacking==[]):
            upd_features_blending = X
            upd_features_stacking = res.T
            X_pca = np.matmul(X,coeff)
            X_pca = np.array(X_pca)
            upd_features_blending_pca = X_pca
        else:
            upd_features_stacking = np.insert(upd_features_stacking,[len(upd_features_stacking[0])],res.T,axis=1)
        upd_features_blending = np.insert(upd_features_blending,[len(upd_features_blending[0])],res.T,axis=1)
        upd_features_blending_pca = np.insert(upd_features_blending_pca,[len(upd_features_blending_pca[0])],res.T,axis=1)

    print('\n Stacking -->')
    print(model3.predict(upd_features_stacking))
    pred_vals.append(model3.predict(upd_features_stacking))

    print('\n Blending -->')
    pred_vals.append(model2.predict(upd_features_blending))
    print(model2.predict(upd_features_blending))

    print('\n PCA -->')

    pred_vals.append(model1.predict(upd_features_blending_pca))
    print(model1.predict(upd_features_blending_pca))
    print(pred_vals)
    return pred_vals
   


'''
Most prominent features contributing towards the price
'''

def prominent_features():
    coeff = (pickle.load(open("saved/coeff",'rb')))

    coeff = np.array(coeff)
    X = np.absolute(coeff.T)

    plist = []
    s = set()
    for i in range(0,X.shape[0]):
        p = X[i].argsort()
        X[i] = np.sort(X[i])
        X[i] = np.flipud(X[i])
        p = list(reversed(p))
        s.add(p[0])
        plist.append(p)



    print('\nThese feature indexes contribute most to new features:-')
    for i in s:
        print(features_to_input[i])
    print('\n')

'''
Main Function
'''   
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    X = np.array(df.drop(['price'],1))
    print('Feature Scaling...')
    X, mean ,std = feature_normalize(X)
    y = np.array(df['price'])

    m = check_accuracy(X,y)
    prominent_features()

    plot_comparison(m)



    X_pred = user_feature(df)
    while(not X_pred==None):
        X_pred = feature_norm_given(X_pred,mean,std)
        pred_vals = predict_user_input(X_pred)
        sum_p = 0
        ct = 0
        for i in pred_vals:
            sum_p = sum_p + float(i)
            ct = ct + 1
        print('\n Average value from all classifiers is \n',sum_p/ct)

        ''' 
        check for more input 
        '''
        X_pred = user_feature(df)
    