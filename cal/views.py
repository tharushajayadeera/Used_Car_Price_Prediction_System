from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def result(request):

    model_year = request.GET['model_year']
    transmission = request.GET['transmission']
    body_type = request.GET['body_type']
    fuel_type = request.GET['fuel_type']
    engine_capacity = request.GET['engine_capacity']
    mileage = request.GET['mileage']

    return render(request, 'result.html',{'result': model_year})



def MLR(request):

    import pandas as pd  
    import numpy as np  
    from sklearn.model_selection import train_test_split 

    cars = pd.read_csv('Online_Dataset.csv')

    X = cars.iloc[:,3:] 
    y = cars.iloc[:,2]

    #Convert the column into categorical columns

    states=pd.get_dummies(X['Fuel_Type'],drop_first=True)

    # Drop the state coulmn
    X=X.drop('Fuel_Type',axis=1)

    # concat the dummy variables
    X=pd.concat([X,states],axis=1)

    #Convert the column into categorical columns
    states=pd.get_dummies(X['Color'],drop_first=True)

    # Drop the state coulmn
    X=X.drop('Color',axis=1)

    # concat the dummy variables
    X=pd.concat([X,states],axis=1)

    X=X.drop(['Cylinders',
            'Age_08_04',
            'Parking_Assistant',
            'Power_Steering',
            'Airbag_1',
            'Airbag_2',
            'Mfg_Month',
            'Met_Color',
            'BOVAG_Guarantee',
            'Mfr_Guarantee',
            'Backseat_Divider',
            'Automatic',
            'HP',
            'Boardcomputer',
            'ABS',
            'Powered_Windows',
            'Mistlamps',
            'Tow_Bar',
            'Sport_Model'
            ], axis=1)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression() 

    regressor.fit(X_train, y_train) #training the algorithm

    y_pred = regressor.predict(X_test)

    from sklearn.metrics import r2_score
    score=r2_score(y_test,y_pred)
    score

    return render(request, 'home.html', {'MLR':score})
