import json
import numpy as np
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import AffinityPropagation
import yfinance as yf

input_file = 'company_symbol_mapping.json'

with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

start_date = '2003-07-03'
end_date = '2007-05-04'

quotes = []
valid_symbols = []
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty and 'Open' in data.columns and 'Close' in data.columns:
            quotes.append(data)
            valid_symbols.append(symbol)
        else:
            print(f"Skipping symbol {symbol} due to missing data.")
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {e}")

symbols = np.array(valid_symbols)
names = np.array([company_symbols_map[symbol] for symbol in symbols])

opening_quotes = np.array([quote['Open'].values for quote in quotes]).astype(np.float64)
closing_quotes = np.array([quote['Close'].values for quote in quotes]).astype(np.float64)

quotes_diff = closing_quotes - opening_quotes

X = quotes_diff.copy().T
X /= X.std(axis=0)

edge_model = GraphicalLassoCV()
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

affinity_model = AffinityPropagation()
affinity_model.fit(edge_model.covariance_)

labels = affinity_model.labels_
num_labels = labels.max()

for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))
