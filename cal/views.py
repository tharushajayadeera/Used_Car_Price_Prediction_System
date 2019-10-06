from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.


def home(request):

    return render(request, 'home.html', {'home':''})


def result(request):

    model_year = request.GET['model_year']
    transmission = request.GET['transmission']
    body_type = request.GET['body_type']
    fuel_type = request.GET['fuel_type']
    engine_capacity = request.GET['engine_capacity']
    mileage = request.GET['mileage']

    hatchback=0
    saloon=0
    stationwagon=0
    petrol=0
    petrolandother=0
    dieselandother=0

    if transmission=="manual": transmission=1 
    if transmission=="automatic": transmission=0

    if body_type=="hatchback": Hatchback=1 
    if body_type=="saloon": Saloon=1 
    if body_type=="stationwagon": stationwagon=1

    if fuel_type=="petrol": petrol=1
    if fuel_type=="diesel": petrol=0
    if fuel_type=="petrolandother": petrolandother=1
    if fuel_type=="dieselandother": dieselandother=1

    import pandas as pd  
    import numpy as np    
    from sklearn.model_selection import train_test_split 
    from sklearn.linear_model import LinearRegression
    
    cars = pd.read_csv('Sri_Lankan_Dataset_Bit_Map_Indexed.csv')
    
    X = cars.iloc[:,1:] 
    y = cars.iloc[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) #training the algorithm
    
    X_test = pd.DataFrame({
           'Model year':[model_year],
           'Engine capacity':[engine_capacity],
           'Mileage':[mileage],
           'Manual':[transmission],
           'Hatchback':[hatchback],
           'Saloon':[saloon],
           'Station wagon':[stationwagon],
           'Diesel, Other fuel type':[dieselandother],
           'Petrol':[petrol],
           'Petrol, Other fuel type':[petrolandother]})

    y_pred = regressor.predict(X_test)  

    return render(request, 'result.html',{'result': y_pred})



