#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brute, fmin
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request
import math


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--download-data',
        action='store_true',
        dest='download_data',
        help='Download fresh data and then run',
        default=False
    )

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
        '--S_0',
        required=False,
        dest='guess_s_0',
        help='S_0. Defaults to 100000',
        metavar='S_0',
        type=int,
        default=15000)

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

    return ( args.download_data, args.start_date, args.predict_range, args.guess_s_0, args.i_0, args.r_0, args.d_0)


def remove_province(input_file, output_file):
    input = open(input_file, "r")
    output = open(output_file, "w")
    output.write(input.readline())
    for line in input:
        if line.lstrip().startswith(","):
            output.write(line)
    input.close()
    output.close()


def download_data(url_dictionary):
    #Lets download the files
    for url_title in url_dictionary.keys():
        urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, loss, start_date, predict_range, guess_s_0, i_0, r_0, d_0):
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.guess_s_0 = guess_s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0




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


    def train(self):
        df = pd.read_csv('data/train.csv')
        rec = pd.read_csv('data/time_series_19-covid-Recovered.csv')
        province_countries = {'Australia',
 'China',
 'Denmark',
 'France',
 'Netherlands',
 'United Kingdom'}
        sub = pd.read_csv('data/submission.csv')
        dict = {}
        for i in range(313):
          country = df.loc[i*84].Country_Region
          province =  df.loc[i*84].Province_State
          print(country, province)
          if country in province_countries:
            confirmed = df[i*84:(i+1)*84].ConfirmedCases
            death = df[i*84:(i+1)*84].Fatalities
            recovered = rec[rec['Country/Region'] == country]
            if isinstance(df.loc[i*84].Province_State, float):
                recovered = recovered[recovered['Province/State'].isnull()]
            else:
                recovered = recovered[recovered['Province/State']==province]
            recovered = recovered.iloc[0].loc[self.start_date:]
            recovered = recovered[:84]
            data = confirmed.values - recovered.values - death.values

            idx  = next((i for i, x in enumerate(confirmed.values) if x), None)
            idx = min(71, idx)
            s0_guess = max(confirmed.values)
            s0_guess = max(s0_guess, 1000)
            bounds=[(s0_guess*2/3, s0_guess*4/3),(0.00001, 0.0001), (0.001, 0.01), (0.001, 0.01)]

            i_0, r_0, d_0 = max(data[idx],1), recovered[idx], death.values[idx]
            #i_0, r_0, d_0 = max(data[idx],1), max(recovered[idx],1), max(death.values[idx],1)
            print(idx, i_0, r_0, d_0)

            rranges = (slice(bounds[0][0], bounds[0][1], s0_guess/5), slice(bounds[1][0], bounds[1][1], 0.00001), slice(bounds[2][0], bounds[2][1], 0.001), slice(bounds[3][0], bounds[3][1], 0.001))
            optimal = brute(loss, rranges, args=(data[idx:], recovered.values[idx:], death.values[idx:], i_0, r_0, d_0), full_output=True, finish=fmin)

            print("optimized s_0, beta, gamma, mu :", optimal[0])
            print("mse:", optimal[1])
            print(optimal[2].shape)

            s_0, beta, gamma, mu = optimal[0]

            if isinstance(df.loc[i*84].Province_State, float):
                dict[country] = [s_0, beta, gamma, mu, optimal[1], optimal[1]/confirmed.values[-1]]
            else:
                dict[country+" "+province] = [s_0, beta, gamma, mu, optimal[1], optimal[1]/confirmed.values[-1]]

            mu = max(mu, 0) # death rate can't be negative
            new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, mu, data[idx:], recovered[idx:], death[idx:], s_0, i_0, r_0, d_0, idx)


            sub.ConfirmedCases[i*43:(i+1)*43] = [round(e) for e in (prediction.y[1][-43:] + prediction.y[2][-43:]+ prediction.y[3][-43:])]
            sub.Fatalities[i*43:(i+1)*43] = [round(e) for e in (prediction.y[3][-43:])]
            plot_df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2], 'Death': prediction.y[3]}, index=new_index)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(country)
            plot_df.plot(ax=ax)
            #print(f"country={country}, s_0={s_0:.8f}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}, mu={mu:.8f}")
            fig.savefig(f"{country}-{province}.png")
        sub = sub.astype(int)
        sub.to_csv('data/submission.csv')
        coef_df = pd.DataFrame.from_dict(dict)
        coef_df.to_csv('data/coef-provinces.csv')




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
    result  = np.sqrt(np.sqrt(np.mean((solution.y[2] - recovered)**2+(solution.y[1] - data)**2+(solution.y[3]-death)**2)))
    return result


def main():

    download, startdate, predict_range , s_0, i_0, r_0, d_0 = parse_arguments()

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)



    learner = Learner(loss, startdate, predict_range, s_0, i_0, r_0, d_0)
    learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')


if __name__ == '__main__':
    main()
