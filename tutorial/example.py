import pandas as pd
import matplotlib.pyplot as plt


def replaceCommaToDot(string):
    return float(string.replace(",", "."))


exchange_rate = pd.read_csv("usd_exchange_rate.csv", delimiter=",")
#exchange_rate['curs'].apply(replaceCommaToDot)
print(exchange_rate.describe())
plt.plot(exchange_rate['curs'])
