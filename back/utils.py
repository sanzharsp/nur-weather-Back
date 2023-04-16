import requests

def numpy_data(location):
    response = requests.post("https://api.openweathermap.org/data/2.5/forecast?{}&appid=45191ceba2adc2581746107474f27a07&units=metric&cnt=5&exclude=hourly,minutely".format(location))    
    return(response.json())
