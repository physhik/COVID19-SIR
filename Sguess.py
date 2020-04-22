import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from tqdm import tqdm
import argparse
from datetime import timedelta, datetime
import matplotlib.pyplot as plt



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
    def __init__(self, start_date, predict_range, idx, i_0, r_0, d_0):
        self.start_date = start_date
        self.predict_range = predict_range
        self.idx = idx
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0
        self.end = True
        self.len_data_4th= 84

    def coef_predict(self, data):
        history = [x for x in data]
        predictions = list()

        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit( disp=0)
        return list(model_fit.forecast(steps=30)[0])

    def train(self):
        df = pd.read_csv('data/train.csv')
        rec = pd.read_csv('data/time_series_19-covid-Recovered.csv')
        province_countries = set(df.Country_Region[df.Province_State.notna()])
        sub = pd.read_csv('data/submission.csv')
        total = df.groupby(['Country_Region','Date']).sum()

        coef = pd.read_csv("data/coef.csv")
        sub = pd.read_csv("data/submission.csv")

        l = []

        for i in range(313):
            if sub.Fatalities[42+43*i]<0:
                l.append(i)

        negative_Fatalities = [sub.Fatalities[42+43*x] for x in l]

        col = coef.columns
        countries = [col[i+1] for i in l]

        for i in range(len(l)):
            coef[countries].loc[0][i] -= [sub.Fatalities[42+43*x] for x in l][i]

        S = coef[countries].loc[0]




        n_areas = 313
        for i in tqdm(range(n_areas)):
            country = df.loc[i*self.len_data_4th].Country_Region
            province =  df.loc[i*self.len_data_4th].Province_State
            confirmed = df[i*self.len_data_4th:(i+1)*self.len_data_4th].ConfirmedCases
            death = df[i*self.len_data_4th:(i+1)*self.len_data_4th].Fatalities
            recovered = rec[rec['Country/Region'] == country]
            if country not in province_countries:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_data_4th]
            elif country in ['US', 'Canada']:
                recovered = recovered.iloc[0].loc[self.start_date:]
                recovered = recovered[:self.len_data_4th]
                total_confirmed = total.loc[country].ConfirmedCases.values
                sr = sum(recovered.values)
                for j in range(self.len_data_4th):
                    recovered[j] = int(confirmed.values[j]*sr/sum(total_confirmed))
            else:
                if isinstance(df.loc[i*self.len_data_4th].Province_State, float):
                    recovered = recovered[recovered['Province/State'].isnull()]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_data_4th]
                else:
                    recovered = recovered[recovered['Province/State']==province]
                    recovered = recovered.iloc[0].loc[self.start_date:]
                    recovered = recovered[:self.len_data_4th]

            if self.idx == None or country == 'China':
                idx  = next((i for i, x in enumerate(confirmed.values) if x), None)
            else:
                idx = max(self.idx, next((i for i, x in enumerate(confirmed.values) if x), None) )

            len_submission = 43
            max_limit_idx = self.len_data_4th- len_submission + self.predict_range
            idx = min(max_limit_idx, idx)

            gr = recovered.values[idx:]
            gc = confirmed.values[idx:]
            gd = death.values[idx:]
            data = gc - gr - gd

            s0 = S[i]
            print(s0, data[0], gr[0], gd[0])
            mu, gamma, s, beta, r0= [], [], [], [], []
            for j in range(len(data)):
                s.append(s0-data[j]-gr[j]-gd[j])

            for j in range(len(data)-1):
                mu.append((gd[j+1]-gd[j])/data[j])
                gamma.append((gr[j+1]-gr[j])/data[j])
                beta.append((s[j]-s[j+1])/(data[j]*s[j]))
                r0.append(((s[j]-s[j+1])/(data[j]*s[j])/((gr[j+1]-gr[j])/data[j])))
            print("predict start")
            predicted_beta = self.coef_predict(beta)
            predicted_gamma = self.coef_predict(gamma)
            predicted_mu = self.coef_predict(mu)
            print("predict done")

            data = list(data)
            dailySIRD = [s[-1], data[-1], gr[-1], gd[-1]]
            updated_dailySIRD = [s[-1], data[-1], gr[-1], gd[-1]]

            predicted_data = []


            for j in tqdm(range(self.predict_range)):
                updated_dailySIRD[0] = dailySIRD[0] - predicted_beta[j] * dailySIRD[1] * dailySIRD[0]
                updated_dailySIRD[1] = dailySIRD[1] + predicted_beta[j] * dailySIRD[1] * dailySIRD[0] - (predicted_gamma[j] + predicted_mu[j]) * dailySIRD[1]
                updated_dailySIRD[2] = dailySIRD[2] + predicted_gamma[j] * dailySIRD[1]
                updated_dailySIRD[3] = dailySIRD[3] + predicted_mu[j] * dailySIRD[1]
                dailySIRD = updated_dailySIRD[:]
                predicted_data.append(updated_dailySIRD[:])


            x = np.array(predicted_data)

            extended_actual =  (data+[None]*self.predict_range)
            predicted_infected = ([None]*len(data)+list(x[:,1]))

            extended_susceptible = (s+[None]*self.predict_range)
            predicted_susceptible = ([None]*len(data)+list(x[:,0]))

            extended_recovered = (list(gr)+[None]*self.predict_range)
            predicted_recovered = ([None]*len(data)+list(x[:,2]))

            extended_death = (list(gd)+[None]*self.predict_range)
            predicted_death = ([None]*len(data)+list(x[:,3]))
            plot_df = pd.DataFrame({'Susceptible': extended_susceptible, 'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Predicted Susceptible': predicted_susceptible, 'Predicted Infected': predicted_infected , 'Predicted Recovered': predicted_recovered, 'Predicted Death': predicted_death})
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(country+", "+str(province))
            plot_df.plot(ax=ax)
            fig.savefig(f"figs/{country}-{province}.png")

def main():

    startdate, predict_range ,idx, i_0, r_0, d_0 = parse_arguments()


    learner = Learner( startdate, predict_range, idx, i_0, r_0, d_0)
    learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')

if __name__ == '__main__':
    main()
