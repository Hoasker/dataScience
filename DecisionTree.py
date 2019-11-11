import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



DTR = DecisionTreeRegressor()

# Constants
past = 7 * 4
future = 7

raw_df = []

money = pd.read_csv("usd_exchange_rate.csv")
values = money["curs"]
# print(values)

start = past
end = len(values) - future
# print(end)


for i in range(start, end):
	# [-28 ... +7]
	past_and_future_values = values[(i-past):(i+future)] 
	raw_df.append(list(past_and_future_values))

# print(raw_df)

past_columns = [f"past_{i}" for i in range(past)]
#print(past_columns)

future_columns = [f"future_{i}" for i in range(future)]
#print(future_columns)

df = pd.DataFrame(raw_df, columns=(past_columns+future_columns))
#print(df)

# Факторы по которым делаем предсказание
x = df[past_columns][:-1]
# То что будем предсказывать
y = df[future_columns][:-1]
# Мульти-регрессия, предсказываем 7 значений

# Тестовая выборка
x_test = df[past_columns][-1:]
y_test = df[future_columns][-1:]


DTR.fit(x,y)

# Предсказание
prediction = DTR.predict(x_test)
print(prediction)

plt.plot(prediction[0], label='prediction')
plt.plot(y_test.iloc[0], label='real')
plt.show()

abs_error = mean_absolute_error(y_test, prediction)
# 0.2614285714285692
print(abs_error)

squared_error = mean_squared_error(y_test, prediction)
# 0.21604285714286092
print(squared_error)