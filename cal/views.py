from django.shortcuts import render
from django.http import HttpResponse
from .models import Algorithm

# Create your views here.

def index(request):

    return render(request, 'home.html',{'index':""})


def home(request):

    algo = Algorithm()
    algo.MLR = MLR(request)
    algo.RFR = RFR(request)
    algo.DTR = DTR(request)
    algo.GBR = GBR(request)

    import numpy as np
    algo.average = np.round(((algo.MLR+algo.RFR+algo.DTR+algo.GBR)/4),0)

    return render(request, 'result.html',{'algo': algo})


def MLR(request):

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

    if body_type=="hatchback": hatchback=1 
    if body_type=="saloon": saloon=1 
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
    y_pred=np.round(y_pred,0)  

    return y_pred


def RFR(request):

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

    if body_type=="hatchback": hatchback=1 
    if body_type=="saloon": saloon=1 
    if body_type=="stationwagon": stationwagon=1

    if fuel_type=="petrol": petrol=1
    if fuel_type=="diesel": petrol=0
    if fuel_type=="petrolandother": petrolandother=1
    if fuel_type=="dieselandother": dieselandother=1

    import pandas as pd  
    import numpy as np    
    from sklearn.model_selection import train_test_split 
    from sklearn.ensemble import RandomForestRegressor
    
    cars = pd.read_csv('Sri_Lankan_Dataset_Bit_Map_Indexed.csv')
    
    X = cars.iloc[:,1:] 
    y = cars.iloc[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)   
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
    y_pred=np.round(y_pred,0) 

    return y_pred


def DTR(request):

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

    if body_type=="hatchback": hatchback=1 
    if body_type=="saloon": saloon=1 
    if body_type=="stationwagon": stationwagon=1

    if fuel_type=="petrol": petrol=1
    if fuel_type=="diesel": petrol=0
    if fuel_type=="petrolandother": petrolandother=1
    if fuel_type=="dieselandother": dieselandother=1

    import pandas as pd  
    import numpy as np    
    from sklearn.model_selection import train_test_split 
    from sklearn.tree import DecisionTreeClassifier
    
    cars = pd.read_csv('Sri_Lankan_Dataset_Bit_Map_Indexed.csv')
    
    X = cars.iloc[:,1:] 
    y = cars.iloc[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = DecisionTreeClassifier()
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
    y_pred=np.round(y_pred,0) 

    return y_pred


def GBR(request):

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

    if body_type=="hatchback": hatchback=1 
    if body_type=="saloon": saloon=1 
    if body_type=="stationwagon": stationwagon=1

    if fuel_type=="petrol": petrol=1
    if fuel_type=="diesel": petrol=0
    if fuel_type=="petrolandother": petrolandother=1
    if fuel_type=="dieselandother": dieselandother=1

    import pandas as pd  
    import numpy as np    
    from sklearn.model_selection import train_test_split 
    from sklearn.ensemble import GradientBoostingRegressor
    
    cars = pd.read_csv('Sri_Lankan_Dataset_Bit_Map_Indexed.csv')
    
    X = cars.iloc[:,1:] 
    y = cars.iloc[:,0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0)
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
    y_pred=np.round(y_pred,0)

    return y_pred
