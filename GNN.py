import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv('common_stops_with_both_names_unique.csv')
data['Date'] = pd.to_datetime(data['Date'])

stop_daily_delays = data.groupby(['Location', 'Date']).agg({
    'Min Delay': 'mean',
    'stop_lat': 'first',
    'stop_lon': 'first'
}).reset_index()

le = LabelEncoder()
stop_daily_delays['stop_id'] = le.fit_transform(stop_daily_delays['Location'])

route_based_edges = data[['Route', 'Location']].drop_duplicates()
edges = []

for route in route_based_edges['Route'].unique():
    stops_on_route = route_based_edges[route_based_edges['Route'] == route]['Location'].values
    for i in range(len(stops_on_route) - 1):
        edges.append((le.transform([stops_on_route[i]])[0], le.transform([stops_on_route[i + 1]])[0]))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

node_features = torch.tensor(stop_daily_delays[['stop_lat', 'stop_lon', 'Min Delay']].values, dtype=torch.float)



class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(3, 64)  # Increase hidden layer size
        self.conv2 = GCNConv(64, 32)  # Second convolution layer
        self.conv3 = GCNConv(32, 16)  # Deeper architecture
        self.conv4 = GCNConv(16, 1)  # Final output layer
        self.dropout = torch.nn.Dropout(p=0.3)  # Add dropout for regularization

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x



data = Data(x=node_features, edge_index=edge_index)

model = GNN()


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

train_data = stop_daily_delays[stop_daily_delays['Date'].dt.month <= 9]
test_data = stop_daily_delays[stop_daily_delays['Date'].dt.month > 9]

train_stop_ids = torch.tensor(train_data['stop_id'].values, dtype=torch.long)
train_labels = torch.tensor(train_data['Min Delay'].values, dtype=torch.float).view(-1, 1)

for epoch in range(400):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    out_train = out[train_stop_ids]

    loss = loss_fn(out_train, train_labels)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

test_stop_ids = torch.tensor(test_data['stop_id'].values, dtype=torch.long)
test_labels = torch.tensor(test_data['Min Delay'].values, dtype=torch.float).view(-1, 1)

model.eval()
pred = model(data.x, data.edge_index)
pred_test = pred[test_stop_ids].numpy().flatten()

test_labels_np = test_labels.numpy().flatten()

mae_per_stop = mean_absolute_error(test_labels_np, pred_test)
rmse_per_stop = np.sqrt(mean_squared_error(test_labels_np, pred_test))

print(f"Mean Absolute Error (MAE): {mae_per_stop}")
print(f"Root Mean Square Error (RMSE): {rmse_per_stop}")

test_data['predictions'] = pred_test


print(test_data[['Location', 'Min Delay', 'predictions']])

