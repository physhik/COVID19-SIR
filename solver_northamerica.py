#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brute, fmin
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import ssl
import urllib.request
import math

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--start-date',
        required=False,
        action='store',
        dest='start_date',
        help='Start date on MM/DD/YY format ... I know ...' +
        'It defaults to first data available 1/22/20',
        metavar='START_DATE',
        type=str,
        default="1/22/20")

    parser.add_argument(
        '--prediction-days',
        required=False,
        dest='predict_range',
        help='Days to predict with the model. Defaults to 150',
        metavar='PREDICT_RANGE',
        type=int,
        default=30)

    parser.add_argument(
        '--idx',
        required=False,
        dest='idx',
        metavar='idx',
        type=int,
        default=None)

    parser.add_argument(
        '--I_0',
        required=False,
        dest='i_0',
        help='I_0. Defaults to 2',
        metavar='I_0',
        type=int,
        default=2)

    parser.add_argument(
        '--R_0',
        required=False,
        dest='r_0',
        help='R_0. Defaults to 0',
        metavar='R_0',
        type=int,
        default=1)

    parser.add_argument(
        '--D_0',
        required=False,
        dest='d_0',
        help='D_0. Defaults to 1',
        metavar='D_0',
        type=int,
        default=1)

    args = parser.parse_args()

    return ( args.start_date, args.predict_range, args.idx, args.i_0, args.r_0, args.d_0)

class Learner(object):
    def __init__(self, loss, start_date, predict_range, idx, i_0, r_0, d_0):
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.idx = idx
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0
        self.end = True

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, mu, data, recovered, death, s_0, i_0, r_0, d_0, idx):
        new_index = self.extend_index(recovered.index, self.predict_range+84-idx)
        size = len(new_index)
        def SIRD(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            return [-beta*S*I, beta*S*I-gamma*I - mu * I , gamma*I, mu*I ]
        extended_actual = np.concatenate((data, [None] * (size - len(data))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIRD, [0, size], [s_0,i_0,r_0, d_0], t_eval=np.arange(0, size, 1))

    def predict_end(self, beta, gamma, mu, data, recovered, death, s_0, idx):
        new_index = self.extend_index(recovered.index, self.predict_range+84-idx-1)
        size = len(new_index)
        def SIRD(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            return [-beta*S*I, beta*S*I-gamma*I - mu * I , gamma*I, mu*I ]
        extended_actual = np.concatenate((data, [None] * (size - len(data))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        s1 = s_0 +data[0] + recovered.values[0] + death.values[0] - data[-1]- recovered.values[-1]- death.values[-1]
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIRD, [84-1, 84-1+self.predict_range], [s1, data[-1],recovered.values[-1], death.values[-1]], t_eval=np.arange(84-1, 84-1+self.predict_range, 1))

    def train(self):
        df = pd.read_csv('data/train.csv')
        rec = pd.read_csv('data/time_series_19-covid-Recovered.csv')
        province_countries = {'Canada','US'}
        sub = pd.read_csv('data/submission.csv')
        total = df.groupby(['Country_Region','Date']).sum()
        dict = {}

        for i in range(313):
          country = df.loc[i*84].Country_Region
          province =  df.loc[i*84].Province_State
          print(country, province)

          if country in province_countries:
            confirmed = df[i*84:(i+1)*84].ConfirmedCases
            death = df[i*84:(i+1)*84].Fatalities
            recovered = rec[rec['Country/Region'] == country]

            recovered = recovered.iloc[0].loc[self.start_date:]
            recovered = recovered[:84]
            total_confirmed = total.loc[country].ConfirmedCases.values
            sr = sum(recovered.values)


            for j in range(84):
                recovered[j] = int(confirmed.values[j]*sr/sum(total_confirmed))
            data = confirmed.values - recovered.values - death.values

            if self.idx ==None:
                idx  = next((i for i, x in enumerate(confirmed.values) if x), None)
            else:
                idx = max(self.idx, next((i for i, x in enumerate(confirmed.values) if x), None))
            idx = min(71, idx)


            s0_guess = max(confirmed.values)
            s0_guess = max(s0_guess, 1000)
            bounds=[(int(s0_guess/2), int(s0_guess*2)),(0.00001, 0.0001), (0.001, 0.01), (0.001, 0.01)]

            i_0, r_0, d_0 = int(max(data[idx],1)), int(recovered[idx]), int(death.values[idx])
            #i_0, r_0, d_0 = max(data[idx],1), max(recovered[idx],1), max(death.values[idx],1)
            print(idx, i_0, r_0, d_0)

            rranges = (slice(bounds[0][0], bounds[0][1], int(s0_guess/2)), slice(bounds[1][0], bounds[1][1], 0.00001), slice(bounds[2][0], bounds[2][1], 0.001), slice(bounds[3][0], bounds[3][1], 0.001))
            optimal = brute(loss, rranges, args=(data[idx:], recovered.values[idx:], death.values[idx:], i_0, r_0, d_0), full_output=True, finish=fmin)

            print("optimized s_0, beta, gamma, mu :", optimal[0])
            print("mse:", optimal[1])
            print(optimal[2].shape)

            s_0, beta, gamma, mu = optimal[0]

            dict[country+" "+province] = [s_0, beta, gamma, mu, optimal[1], optimal[1]/confirmed.values[-1]]


            mu = max(mu, 0) # death rate can't be negative
            if self.end:
                new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict_end(beta, gamma, mu, data[idx:], recovered[idx:], death[idx:], s_0, idx)
                sub.ConfirmedCases[i*43 + 13:(i+1)*43] = [round(e) for e in (prediction.y[1] + prediction.y[2]+ prediction.y[3])]
                sub.Fatalities[i*43 + 13:(i+1)*43] = [round(e) for e in (prediction.y[3])]
            else:
                new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, mu, data[idx:], recovered[idx:], death[idx:], s_0, i_0, r_0, d_0, idx)
                sub.ConfirmedCases[i*43:(i+1)*43] = [round(e) for e in (prediction.y[1][-43:] + prediction.y[2][-43:]+ prediction.y[3][-43:])]
                sub.Fatalities[i*43:(i+1)*43] = [round(e) for e in (prediction.y[3][-43:])]
            size = len(new_index)
            predicted_susceptible = np.concatenate(([None] * (size-len(prediction.y[0])), prediction.y[0]))
            predicted_infected = np.concatenate(([None] * ( size-len(prediction.y[1])), prediction.y[1]))
            predicted_recovered = np.concatenate(([None] * ( size-len(prediction.y[2])), prediction.y[2]))
            predicted_death = np.concatenate(([None] * ( size-len(prediction.y[3])), prediction.y[3]))
            plot_df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': predicted_susceptible, 'Infected': predicted_infected , 'Recovered': predicted_recovered, 'Death': predicted_death}, index=new_index)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(country)
            plot_df.plot(ax=ax)
            #print(f"country={country}, s_0={s_0:.8f}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}, mu={mu:.8f}")
            fig.savefig(f"{country}-{province}.png")
        sub = sub.astype(int)
        sub.to_csv(f'data/submission.csv')
        coef_df = pd.DataFrame.from_dict(dict)
        coef_df.to_csv(f'data/coef-Namerica.csv')

def loss(point, data, recovered, death, i_0, r_0, d_0):
    size = len(data)
    s_0, beta, gamma, mu = point
    def SIRD(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        D = y[3]
        return [-beta*S*I, beta*S*I-gamma*I - mu*I , gamma*I, mu*I ]
    solution = solve_ivp(SIRD, [0, size], [s_0,i_0,r_0,d_0], t_eval=np.arange(0, size, 1), vectorized=True)
    W = 0.8
    X = [(solution.y[1] - data), W*(solution.y[2] - recovered), (solution.y[3] - death)]
    mid = size//3

    result = 0
    for x in X:
        Y = [x[:mid], x[mid:mid*2], x[2*mid:]]
        result += sum([sum(Y[w]**2*(w+1)) for w in range(len(Y))])/6/size
    result = np.sqrt(np.sqrt(result))
    return result

def main():

    startdate, predict_range , idx, i_0, r_0, d_0 = parse_arguments()


    learner = Learner(loss, startdate, predict_range, idx, i_0, r_0, d_0)
    learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')

if __name__ == '__main__':
    main()
