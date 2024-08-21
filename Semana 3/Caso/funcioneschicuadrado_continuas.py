# -*- coding: utf-8 -*-
"""FuncionesChiCuadrado_Continuas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WJAA0HzyqwcOAqkIc3mSPpGSN9YMtfRJ

**Librerías**
"""

import math
import numpy as np
import statistics as st
from scipy.stats import expon, norm, lognorm, gamma, weibull_min, beta, uniform, chi2, triang
import pandas as pd

"""**Función Chi-Cuadrado para una distribución normal**"""

def chi_square_normal_goodness_of_fit(data,media="estimado",desvesta="estimado",r=2):
    """Performs a Chi-square goodness of fit test for a normal probability distribution.

    Arguments:
    data -- a list of data values
    media -- Data mean
    desvesta -- Data Standard Deviation
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = "+str(mean))
    else:
        mean = media
    if desvesta=="estimado":
        std_dev = np.std(data)
        print("Parámetro estimado: Desviación Estándar = "+str(std_dev))
    else:
        std_dev = desvesta
    
    n = len(data)
    # Divide the data into k intervals using the mean and standard deviation

    k = math.floor(n/5)

    #Define range limits for equally probable classes

    intervals = np.zeros(k+1)

    for i in range(k+1):
        intervals[i]=norm.ppf((i)/k, loc=mean, scale=std_dev)

    # Calculate the expected frequencies for each interval

    expected_frequencies = np.zeros(k)
    expected_frequencies = [n * (norm.cdf(intervals[i+1], mean, std_dev) - norm.cdf(intervals[i], mean, std_dev)) for i in range(k)]

    # Calculate the observed frequencies for each interval

    observed_frequencies = np.zeros(k)
    observed_frequencies[0]=float(sum(data<=intervals[1]))
    for i in range(1,k):
        observed_frequencies[i]=float(sum((data>=intervals[i]) & (data<=intervals[i+1])))

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k - 1))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)

    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución lognormal**"""

def chi_square_lognormal_goodness_of_fit(data,media="estimado",desvesta="estimado",r=2):
    """Performs a Chi-square goodness of fit test for a lognormal probability distribution with equiprobable classes.

    Arguments:
    data -- a list of data values
    media -- Data mean (lognormal asociada)
    desvesta -- Data Standard Deviation (lognormal asociada)
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(np.log(data))
        print("Parámetro estimado (lognormal asociada): Media = "+str(mean))
    else:
        mean = media
    
    if desvesta=="estimado":
        std_dev = np.std(np.log(data))
        print("Parámetro estimado (lognormal asociada): Desviación Estándar = "+str(std_dev))
    else:
        std_dev = desvesta  
    
    n = len(data)
    k = math.floor(n/5)

    # Divide the data into k equiprobable intervals
    intervals = np.linspace(0, 1, k+1)
    log_intervals = lognorm.ppf(intervals, std_dev, scale=np.exp(mean))

    # Calculate the expected frequencies for each interval
    expected_frequencies = np.diff(n * lognorm.cdf(log_intervals, std_dev, scale=np.exp(mean)))

    # Calculate the observed frequencies for each interval
    observed_frequencies, _ = np.histogram(data, log_intervals)

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución exponencial**"""

def chi_square_exponential_goodness_of_fit(data,tasa="estimado",r=1):
    """Performs a Chi-square goodness of fit test for an exponential probability distribution.

    Arguments:
    data -- a list of data values
    tasa -- rate
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if tasa=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Tasa = "+str(1/mean))
    else:
        mean = 1/tasa
    
    n = len(data)

    # Divide the data into k intervals using the mean

    k = math.floor(n/5)

    intervals = np.zeros(k+1)
    intervals[0] = 0

    for i in range(1,k):
        intervals[i]=-np.log(1-i/k)*mean

    intervals[k]=np.inf

    # Calculate the expected frequencies for each interval
    expected_frequencies = np.zeros(k)
    expected_frequencies = [n * (expon.cdf(intervals[i+1], scale=mean) - expon.cdf(intervals[i], scale=mean)) for i in range(k)]

    # Calculate the observed frequencies for each interval
    observed_frequencies = np.zeros(k)
    observed_frequencies[0]=float(sum(data<=intervals[1]))

    for i in range(1,k):
        observed_frequencies[i]=float(sum((data>=intervals[i]) & (data<=intervals[i+1])))

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k - 1))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)

    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución uniforme**"""

def chi_square_uniform_goodness_of_fit(data,minimo="estimado",maximo="estimado",r=2):
    """Performs a Chi-square goodness of fit test for a uniform probability distribution.

    Arguments:
    data -- a list of data values
    a -- the lower bound of the uniform distribution
    b -- the upper bound of the uniform distribution
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
        print("Parámetro estimado: a = "+str(a))
    else:
        a = minimo
    
    if maximo=="estimado":
        b = np.max(data)
        print("Parámetro estimado: b = "+str(b))
    else:
        b = maximo

    # Divide the data into k intervals

    k = math.floor(n/5)
    # k = math.ceil(np.sqrt(n))

    intervals = np.zeros(k+1)
    intervals = np.linspace(a, b, k+1)

    # Calculate the expected frequencies for each interval
    expected_frequencies = np.zeros(k)
    expected_frequencies = [n * (uniform.cdf(intervals[i+1], loc=a, scale=b-a) - uniform.cdf(intervals[i], loc=a, scale=b-a)) for i in range(k)]

    # Calculate the observed frequencies for each interval
    observed_frequencies = np.zeros(k)
    observed_frequencies, _ = np.histogram(data, intervals)

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k - 1))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución triangular**"""

def chi_square_triangular_goodness_of_fit(data,minimo="estimado",maximo="estimado",moda="estimado",r=3):
    """Performs a Chi-square goodness of fit test for a triangular probability distribution.

    Arguments:
    data -- a list of data values
    a -- the lower bound of the triangular distribution
    b -- the upper bound of the triangular distribution
    c -- the mode of the triangular distribution
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic, p-value, intervals, expected frequencies, and observed frequencies.
    """
    n = len(data)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if minimo=="estimado":
        a = np.min(data)
        print("Parámetro estimado: a = "+str(a))
    else:
        a = minimo

    if maximo=="estimado":
        b = np.max(data)
        print("Parámetro estimado: b = "+str(b))
    else:
        b = maximo
        
    if moda=="estimado":
        c = st.mode(data)
        print("Parámetro estimado: c = "+str(c))
    else:
        c = moda

    # Divide the data into k intervals
    k = math.floor(n / 5)
    intervals = np.linspace(a, b, k + 1)

    # Calculate the expected frequencies for each interval
    expected_frequencies = np.zeros(k)
    expected_frequencies = [n * (triang.cdf(intervals[i + 1], c=(c - a) / (b - a), loc=a, scale=b - a) - triang.cdf(intervals[i], c=(c - a) / (b - a), loc=a, scale=b - a)) for i in range(k)]

    # Calculate the observed frequencies for each interval
    observed_frequencies, _ = np.histogram(data, intervals)

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)

    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución gamma**"""

def chi_square_gamma_goodness_of_fit(data,media="estimado",varianza="estimado",r=2):
    """Performs a Chi-square goodness of fit test for a gamma probability distribution.

    Arguments:
    data -- a list of data values
    media -- Data mean
    varianza -- Data Variance
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if media=="estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = "+str(mean))
    else:
        mean = media
    
    if varianza=="estimado":
        var = np.var(data)
        print("Parámetro estimado: Varianza = "+str(var))
    else:
        var = varianza
    
    n = len(data)

    # Divide the data into k intervals using the mean and variance

    k = math.floor(n/5)

    intervals = np.linspace(0, k*mean, k+1)

    # Calculate the expected frequencies for each interval
    expected_frequencies = [n * (gamma.cdf(intervals[i+1], a=var/mean, scale=mean) - gamma.cdf(intervals[i], a=var/mean, scale=mean)) for i in range(k)]

    # Calculate the observed frequencies for each interval
    observed_frequencies, _ = np.histogram(data, intervals)

    # Calculate the test statistic
    chi_squared = sum((observed_frequencies[i] - expected_frequencies[i])**2 / expected_frequencies[i] for i in range(k-1))

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)
    
    return print(respuesta)

"""**Función Chi-Cuadrado para una distribución weibull**"""

def chi_square_weibull_goodness_of_fit(data,forma="estimado",escala="estimado",r=2):
    """Performs a Chi-square goodness of fit test for a Weibull probability distribution with equiprobable classes.

    Arguments:
    data -- a list of data values
    forma -- Data shape factor
    escala -- Data scale factor
    r -- number of estimated parameters

    Returns:
    A tuple containing the test statistic and p-value.
    """
    shape, loc, scale = weibull_min.fit(data, floc=0)
    # Se verifica si se estiman parámetros o se utilizan los parámetros dados por el usuario
    if escala!="estimado":
        scale = escala
    else:
        print("Parámetro estimado: Escala = "+str(scale))
    
    if forma!="estimado":
        shape = forma
    else:
        print("Parámetro estimado: Forma = "+str(shape))
    
    n = len(data)
    k = math.floor(n/5)  # Number of equiprobable intervals
    intervals = np.linspace(0, np.max(data), k+1)  # Define the intervals

    # Calculate the expected frequencies for each interval
    shape, loc, scale = weibull_min.fit(data, floc=0)
    expected_frequencies = n * np.diff(weibull_min.cdf(intervals, shape, loc=loc, scale=scale))

    # Calculate the observed frequencies for each interval
    observed_frequencies, _ = np.histogram(data, intervals)

    # Calculate the test statistic
    chi_squared = np.sum((observed_frequencies - expected_frequencies)**2 / expected_frequencies)

    # Calculate the degrees of freedom
    degrees_of_freedom = k - r - 1  # Two parameters for shape and scale

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

    respuesta = "Chi-squared statistic: "+ str(chi_squared) + "\np-value: " + str(p_value)

    return print(respuesta)