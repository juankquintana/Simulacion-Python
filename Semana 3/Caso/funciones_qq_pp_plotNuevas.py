import math
import numpy as np
import statistics as st
from scipy.stats import expon, norm, lognorm, gamma, weibull_min, uniform, triang
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def PP_QQ_plot_normal(data, media="estimado", desvesta="estimado"):
    if media == "estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = " + str(mean))
    else:
        mean = media

    if desvesta == "estimado":
        std_dev = np.std(data)
        print("Parámetro estimado: Desviación Estándar = " + str(std_dev))
    else:
        std_dev = desvesta

    n = len(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, norm, loc=mean, scale=std_dev, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(norm.cdf(data, loc=mean, scale=std_dev))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_lognormal(data, media="estimado", desvesta="estimado"):
    if media == "estimado":
        mean = np.mean(np.log(data))
        print("Parámetro estimado (normal asociada): Media = " + str(mean))
    else:
        mean = media

    if desvesta == "estimado":
        std_dev = np.std(np.log(data))
        print("Parámetro estimado (normal asociada): Desviación Estándar = " + str(std_dev))
    else:
        std_dev = desvesta

    n = len(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, lognorm, distargs=(std_dev,), scale=np.exp(mean), line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(lognorm.cdf(data, s=std_dev, scale=np.exp(mean)))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_exponential(data, tasa="estimado"):
    if tasa == "estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Tasa = " + str(1 / mean))
    else:
        mean = 1 / tasa

    n = len(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, expon, scale=mean, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(expon.cdf(data, scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_uniform(data, minimo="estimado", maximo="estimado"):
    n = len(data)
    if minimo == "estimado":
        a = np.min(data)
        print("Parámetro estimado: a = " + str(a))
    else:
        a = minimo

    if maximo == "estimado":
        b = np.max(data)
        print("Parámetro estimado: b = " + str(b))
    else:
        b = maximo
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, uniform, loc=a, scale=b-a, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(uniform.cdf(data, loc=a, scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_triangular(data, minimo="estimado", maximo="estimado", moda="estimado"):
    n = len(data)
    if minimo == "estimado":
        a = np.min(data)
        print("Parámetro estimado: a = " + str(a))
    else:
        a = minimo

    if maximo == "estimado":
        b = np.max(data)
        print("Parámetro estimado: b = " + str(b))
    else:
        b = maximo

    if moda == "estimado":
        c = st.mode(data)
        print("Parámetro estimado: c = " + str(c))
    else:
        c = moda
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, triang, distargs=((c - a)/(b - a),), loc=a, scale=b-a, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(triang.cdf(data, c=(c - a)/(b - a), loc=a, scale=b-a))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_gamma(data, media="estimado", varianza="estimado"):
    if media == "estimado":
        mean = np.mean(data)
        print("Parámetro estimado: Media = " + str(mean))
    else:
        mean = media

    if varianza == "estimado":
        var = np.var(data)
        print("Parámetro estimado: Varianza = " + str(var))
    else:
        var = varianza

    n = len(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, gamma, distargs=(var/mean,), scale=mean, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(gamma.cdf(data, a=var/mean, scale=mean))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()

def PP_QQ_plot_weibull(data, forma="estimado", escala="estimado"):
    n = len(data)
    if forma == "estimado":
        c = (np.std(data) / np.mean(data))**(-1.086)
        print("Parámetro estimado: Forma = " + str(c))
    else:
        c = forma

    if escala == "estimado":
        scale = np.mean(data) / math.gamma(1 + 1/c)
        print("Parámetro estimado: Escala = " + str(scale))
    else:
        scale = escala
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sm.qqplot(data, weibull_min, distargs=(c,), scale=scale, line='45', ax=axes[0])
    axes[0].set_title("Q-Q Plot")
    
    p = np.arange(1, n + 1) / n - 0.5 / n
    pp = np.sort(weibull_min.cdf(data, c=c, scale=scale))
    sns.scatterplot(x=pp, y=p, color='blue', edgecolor='blue', ax=axes[1])
    axes[1].plot(np.linspace(0, 1.01), np.linspace(0, 1.01), 'r', lw=2)
    axes[1].set_title('P-P plot')
    axes[1].set_xlabel('Theoretical Probabilities')
    axes[1].set_ylabel('Sample Probabilities')
    
    fig.tight_layout()
    plt.show()
