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
import time
#from multiprocessing import Pool
from tqdm import tqdm
import ray
import gc
import math

ray.init()

def parse_arguments():
    parser = argparse.ArgumentParser()



    parser.add_argument(
        '--idx',
        required=False,
        dest='idx',
        metavar='idx',
        type=int,
        default=None)

    args = parser.parse_args()

    return (  args.idx )

class Learner(object):
    def __init__(self, loss, idx):
        self.loss = loss
        self.start_date = '1/22/20'
        self.overlap = 20
        self.predict_range = 43 - self.overlap
        self.idx = idx
        self.end = True
        self.len_train = 84

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, mu, data, recovered, death, s_0, i_0, r_0, d_0, idx, n_each):
        new_index = self.extend_index(recovered.index, self.predict_range+n_each-idx)
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

    def predict_end(self, beta, gamma, mu, data, recovered, death, s_0, idx, n_each):
        new_index = self.extend_index(recovered.index, self.predict_range+n_each-idx-1)
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
        s1 = s_0 + data[0] + recovered.values[0] + death.values[0] - data[-1]- recovered.values[-1]- death.values[-1]
        s1 = np.abs(s1)
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIRD, [n_each-1, n_each-1+self.predict_range], [s1, data[-1],recovered.values[-1], death.values[-1]], t_eval=np.arange(n_each-1, n_each-1+self.predict_range, 1))

    def train(self):
        df = pd.read_csv('data/train.csv')
        rec = pd.read_csv('data/time_series_19-covid-Recovered.csv')
        province_countries = set(df.Country_Region[df.Province_State.notna()])
        sub = pd.read_csv('data/submission.csv')
        total = df.groupby(['Country_Region','Date']).sum()
        dict = {}
        Args = []
        n_areas = 8
        n_each = int(len(df)/313)
        for i in range(n_areas):
            country = df.loc[i*n_each].Country_Region
            province =  df.loc[i*n_each].Province_State
            confirmed = df[i*n_each:(i+1)*n_each].ConfirmedCases[:self.len_train ]
            death = df[i*n_each:(i+1)*n_each].Fatalities[:self.len_train ]
            recovered = rec[rec['Country/Region'] == country]
            if country not in province_countries:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_train ]
            elif country in ['US', 'Canada']:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_train ]
                total_confirmed = total.loc[country].ConfirmedCases.values[:self.len_train ]
                sr = sum(recovered.values)
                for j in range(self.len_train):
                    recovered[j] = confirmed.values[j]*sr/sum(total_confirmed)
            else:
                if isinstance(df.loc[i*n_each].Province_State, float):
                    recovered = recovered[recovered['Province/State'].isnull()]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_train ]
                else:
                    recovered = recovered[recovered['Province/State']==province]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_train ]

            data = confirmed.values - recovered.values - death.values

            if self.idx == None or country == 'China':
                idx  = next((i for i, x in enumerate(confirmed.values) if x), None)
            else:
                idx = max(self.idx, next((i for i, x in enumerate(confirmed.values) if x), None) )

            len_submission = 43
            max_limit_idx = n_each- len_submission + self.predict_range
            idx = min(max_limit_idx, idx) #len_data_4th-13 = n_each- len_submission + predict_range = 84 - len_submission + 30 = 71

            s0_guess = max(confirmed.values)
            if s0_guess > 100000:
                h_limit = s0_guess*2
                l_limit = s0_guess/2
                s0_grid_size = s0_guess/3
            elif s0_guess < 2000:
                h_limit = s0_guess * 5
                l_limit = s0_guess/2
                s0_grid_size = s0_guess
            else:
                h_limit = s0_guess * 4
                l_limit = s0_guess/2
                s0_grid_size = s0_guess
            bounds=[(l_limit, h_limit),(0.00001, 0.0001), (0.001, 0.01), (0.001, 0.01)]

            i_0, r_0, d_0 = data[idx], recovered[idx], death.values[idx]
            #i_0, r_0, d_0 = max(data[idx],1), max(recovered[idx],1), max(death.values[idx],1)

            rranges = (slice(bounds[0][0], bounds[0][1], s0_guess), slice(bounds[1][0], bounds[1][1], 0.00001), slice(bounds[2][0], bounds[2][1], 0.001), slice(bounds[3][0], bounds[3][1], 0.001))
            brute_args = (loss, rranges, (data[idx:], recovered.values[idx:], death.values[idx:], i_0, r_0, d_0))
            Args.append(Brute.remote(brute_args))
            #Args.append(brute_args)

        Optimal = ray.get(Args)

        #Optimal = []
        #for i in tqdm(range(40)):
            #with Pool() as p:
                #start = time.time()
                #n = 8
                #optimal = p.map(Brute, Args[n*i:min((i+1)*n, n_areas)])
                #end = time.time()
                #print("brute time: ", end-start)
                #Optimal += optimal


        for i in tqdm(range(n_areas)):
            country = df.loc[i*n_each].Country_Region
            province =  df.loc[i*n_each].Province_State
            confirmed = df[i*n_each:(i+1)*n_each].ConfirmedCases[:self.len_train ]
            death = df[i*n_each:(i+1)*n_each].Fatalities[:self.len_train ]
            recovered = rec[rec['Country/Region'] == country]
            if country not in province_countries:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_train ]
            elif country in ['US', 'Canada']:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_train ]
                total_confirmed = total.loc[country].ConfirmedCases.values[:self.len_train ]
                sr = sum(recovered.values)
                for j in range(self.len_train):
                    recovered[j] = confirmed.values[j]*sr/sum(total_confirmed)
            else:
                if isinstance(df.loc[i*n_each].Province_State, float):
                    recovered = recovered[recovered['Province/State'].isnull()]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_train ]
                else:
                    recovered = recovered[recovered['Province/State']==province]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_train ]

            data = confirmed.values - recovered.values - death.values

            if self.idx == None or country == 'China':
                idx  = next((i for i, x in enumerate(confirmed.values) if x), None)
            else:
                idx = max(self.idx, next((i for i, x in enumerate(confirmed.values) if x), None) )

            len_submission = 43
            max_limit_idx = n_each- len_submission + self.predict_range
            idx = min(max_limit_idx, idx) #len_data_4th-13 = n_each- len_submission + predict_range = 84 - len_submission + 30 = 71

            optimal = Optimal[i]
            s_0, beta, gamma, mu = optimal[0]
            #optimal = brute(loss, rranges, args=(data[idx:], recovered.values[idx:], death.values[idx:], i_0, r_0, d_0), full_output=True, finish=fmin)
            #print("optimized s_0, beta, gamma, mu :", optimal[0])
            #print("mse:", optimal[1])
            #print(optimal[2].shape)
            country = df.loc[i*n_each].Country_Region
            province = str(df.loc[i*n_each].Province_State)
            dict[country+" "+province] = [s_0, beta, gamma, mu, optimal[1], optimal[1]/confirmed.values[-1]]

            #mu = max(mu, 0) # death rate can't be negative
            #start = time.time()
            if self.end:
                new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict_end(beta, gamma, mu, data[idx:], recovered[idx:], death[idx:], s_0, idx, n_each)
                sub.ConfirmedCases[i*len_submission + self.overlap:(i+1)*len_submission] = [round(e) for e in (prediction.y[1] + prediction.y[2]+ prediction.y[3])]
                sub.Fatalities[i*len_submission + self.overlap:(i+1)*len_submission] = [round(e) for e in (prediction.y[3])]
            else:
                new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, mu, data[idx:], recovered[idx:], death[idx:], s_0, i_0, r_0, d_0, idx)
                sub.ConfirmedCases[i*len_submission:(i+1)*len_submission] = [round(e) for e in (prediction.y[1][-len_submission:] + prediction.y[2][-len_submission:]+ prediction.y[3][-len_submission:])]
                sub.Fatalities[i*len_submission:(i+1)*len_submission] = [round(e) for e in (prediction.y[3][-len_submission:])]
            #end = time.time()
            #print("predict time: ", end-start)
            size = len(new_index)
            predicted_susceptible = np.concatenate(([None] * (size-len(prediction.y[0])), prediction.y[0]))
            predicted_infected = np.concatenate(([None] * ( size-len(prediction.y[1])), prediction.y[1]))
            predicted_recovered = np.concatenate(([None] * ( size-len(prediction.y[2])), prediction.y[2]))
            predicted_death = np.concatenate(([None] * ( size-len(prediction.y[3])), prediction.y[3]))
            plot_df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': predicted_susceptible, 'Infected': predicted_infected , 'Recovered': predicted_recovered, 'Death': predicted_death}, index=new_index)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(country+", "+province)
            plot_df.plot(ax=ax)
            print(f"country={country}, s_0={s_0:.8f}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}, mu={mu:.8f}")
            fig.savefig(f"figs/{country}-{province}.png")
        sub = sub.astype(int)
        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]
        train = pd.read_csv('data/train.csv')
        for i in range(313):
            sub.ConfirmedCases[43*i:self.overlap+43*i] = list(train.ConfirmedCases.values[(i+1)*n_each-self.overlap:(i+1)*n_each])
            sub.Fatalities[43*i:self.overlap+43*i] = list(train.Fatalities.values[(i+1)*n_each-self.overlap:(i+1)*n_each])
        sub.to_csv(f'data/submission.csv')
        coef_df = pd.DataFrame.from_dict(dict)
        coef_df.to_csv(f'data/coef.csv')

@ray.remote
def Brute(Args):
    f, x, a = Args
    optimal = brute(f, x, args = a, full_output=True, finish=fmin)
    return optimal

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
    #Y = [x for x in data] +[x for x in recovered]+ [x for x in death]
    #Y_pred = [x for x in solution.y[1]] + [x for x in solution.y[2]]+ [x for x in solution.y[3]]
    #result = rmsle(Y, Y_pred) + max(0, -s_0)
    W = 0.8
    X = [(solution.y[1] - data), W*(solution.y[2] - recovered), (solution.y[3] - death)]
    #mid = size//6
    mid = size//3

    result = 0
    for x in X:
        Y = [x[:mid], x[mid:mid*2], x[2*mid:]]
        result += sum([sum(Y[w]**2*(w+1)) for w in range(len(Y))])/6/size
        #Y = [x[:mid], x[mid:mid*2], x[2*mid:3*mid], x[3*mid:4*mid], x[4*mid:5*mid], x[5*mid:]]
        #result += sum([sum(Y[w]**2*(w+1)**1.5) for w in range(len(Y))])/43/size
    result = np.sqrt(np.sqrt(result))
    return result


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    #y, y_pred = list(y), list(y_pred)
    terms_to_sum = 0
    for i,pred in enumerate(y_pred):
        if y[i] != 0:
            if pred < 0:
                terms_to_sum += 10
            else:
                terms_to_sum += (math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0
        elif y[i] == 0:
            if pred !=0:
                terms_to_sum += 10

    return (terms_to_sum * (1.0/len(y)))**0.5

def main():

    idx = parse_arguments()


    learner = Learner(loss, idx)
    learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')

if __name__ == '__main__':
    main()
