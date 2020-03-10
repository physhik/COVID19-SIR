import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

S_0 = 15000
I_0 = 2
R_0 = 0

START_DATE = {
  'Japan': '1/22/20',
  'Italy': '1/31/20',
  'South Korea': '1/22/20',
  'Iran': '2/19/20'
}

def load_confirmed(country):
  df = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
  country_df = df[df['Country/Region'] == country]
  return country_df.iloc[0].loc[START_DATE[country]:]

def loss(point, data):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l = np.mean((solution.y[1] - data)**2)
    print(f"point={point}, loss={l}")
    return l

def extend_index(index, new_size):
    values = index.values
    current = datetime.strptime(index[-1], '%m/%d/%y')
    while len(values) < new_size:
        current = current + timedelta(days=1)
        values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
    return values

def plot(beta, gamma, data, country):
    predict_range = 150
    new_index = extend_index(data.index, predict_range)
    size = len(new_index)
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))

    extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
    df = pd.DataFrame({'Actual': extended_actual, 'S': solution.y[0], 'I': solution.y[1], 'R': solution.y[2]}, index=new_index)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(country)
    df.plot(ax=ax)
    fig.savefig(f"{country}.png")

def predict(country):
    data = load_confirmed(country)
    optimal = minimize(loss, [0.001, 0.001], args=(data), method='L-BFGS-B', bounds=[(0.0, 0.1), (0.0, 0.1)])
    print(optimal)
    beta, gamma = optimal.x
    print(f"R_0={beta*100/gamma}%")
    plot(beta, gamma, data, country)


# predict('Japan')
# predict('South Korea')
# predict('Italy')
predict('Iran')
