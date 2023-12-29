# Churn-Rate-Prediction
Build model machine learning for Churn Rate Prediction. 

## Setups
```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install -U catboost
```
## Prepare Data

1. Load data:
```python
path = 'data\churn_rate_prediction.csv'
df = pd.read_csv(path)
```
2. Data Analysis
   - Numerical
![image](data/Screenshot%202023-12-29%20093243.png)
   - Categorical
![image1](data/Screenshot%202023-12-29%20093518.png)
   - Relationship between features with the target
![image2](data/Screenshot%202023-12-29%20094802.png)
3. Feature Engineering
   - Remove columns unnecessary
   - Format data
```python
df.drop(['customer_id' ,'Name' ,  'security_no' , 'referral_id' , 'last_visit_time'] , axis = 1 , inplace = True)
# transform join_date -> days_since_joined
df['joining_date'] = df['joining_date'].astype('datetime64[ns]')
df['days_since_joined'] = df['joining_date'].apply(lambda x: (pd.Timestamp('today')-x).days)
df.drop(columns = 'joining_date' , inplace = True)
df.replace(['?' , 'Error'] , np.NaN , inplace = True)
def as_type(data):
  columns = data.columns
  for i in columns:
    if data[i].dtypes == 'O':
      try:
        data[i] = data[i].astype('float64')
        print('Sucessfull')
      except:
        print('lose')
  return data
df = as_type(df)
```
  - Fill missing value for NaN
  - Apply statistics for data
  - Convert string to numeric
```python
def Apply_feature_Engineer(data , features , features_o , categorical):
  for i in features:
    data.loc[data[i] < 0 , i] = np.NaN
  for i in features_o:
    Q = data[i].quantile([0.95 , 0.05])
    data.loc[data[i] > Q[0.95] , i] = Q[0.95]
    data.loc[data[i] < Q[0.05] , i] = Q[0.05]

  data = pd.get_dummies(data , columns = categorical )
  return data
```
## Build model: 
1. Decision Tree:
```python
def build_model(X, y):
    clf = DecisionTreeClassifier()
    scaler = StandardScaler()
    # handle missing value
    Iter = IterativeImputer()
    estimators = [
    BayesianRidge(),
    #KNeighborsRegressor(n_neighbors=5),
    ]
    # BayesianRidge better than KNeighborsRegressor
    param_grid = {
    'Iter__estimator':estimators,
    'tree__criterion': ["gini", "entropy", "log_loss"],
    }
    pipe = Pipeline([('scaler', scaler),('Iter' , Iter),("tree", clf)])
    model = GridSearchCV(pipe, param_grid, scoring="recall", n_jobs=-1)
    model.fit(X,y)

    return model
```
- Evaluated
![Evaluated1](data/Screenshot%202023-12-29%20102934.png)

1. Random Forest:
```python
def model_RandomForest(X , Y):
    RFC = RandomForestClassifier()
    scaler = StandardScaler()
    n_estimators = [80 , 90 , 100]
    simple = SimpleImputer()
    strategys = ["median"  ,"mean", 'most_frequent']

    param_grid = {
    'simple__strategy':strategys,
    'tree__n_estimators': n_estimators,
    }


    # Todo: Input your scaler and logistic model into pipeline
    pipe = Pipeline([('simple' ,simple),('scaler', scaler),("tree", RFC )])
    # Todo: fit your model with X, y
    model =GridSearchCV(pipe, param_grid, n_jobs=-1 ,scoring="recall" )
    model.fit(X,Y)
    return model
```
- Evaluated

![Evaluated2](data/Screenshot%202023-12-29%20103748.png)
3. CatBoost:
```python
def build_model_RF_Iter(X, y):
    clf = CatBoostClassifier()
    scaler = StandardScaler()
    n_estimators = [50 , 60 , 70]
    Iter = IterativeImputer()
    estimators = [
    BayesianRidge(),
    #KNeighborsRegressor(n_neighbors=5),
    ]

    param_grid = {
    'Iter__estimator':estimators,
    'tree__n_estimators': n_estimators,
    }

    # Todo: Input your scaler and logistic model into pipeline
    pipe = Pipeline([('Iter' , Iter),('scaler', scaler),("tree", clf)])
    model = GridSearchCV(pipe, param_grid, scoring="accuracy", n_jobs=-1 )
    # Todo: fit your model with X, y
    model.fit(X,y)

    return model
```
![Evaluated3](data/Screenshot%202023-12-29%20104311.png)

4. SVM:
```python
def build_model_LSVM_SimpleImputer(X, y):
    svm = LinearSVC(random_state = 0)
    scaler = StandardScaler()
    C = [0.2 , 0.5 , 1]
    #Iter = IterativeImputer(KNeighborsRegressor(n_neighbors= 30))
    param_grid = {
        #'svm__penalty':penalty,
        'svm__C' : C
    }

    # Todo: Input your scaler and logistic model into pipeline
    pipe = Pipeline([('scaler', scaler),("svm", svm)])
    model = GridSearchCV(pipe, param_grid, scoring="recall", n_jobs=-1)
    # Todo: fit your model with X, y
    model.fit(X,y)

    return model
```
![Evaluated4](data/Screenshot%202023-12-29%20104736.png)