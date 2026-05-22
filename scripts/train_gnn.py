import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import sys

# Ensure pure root routing securely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.gnn_model import GNNFraudDetector, MODEL_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def synthesize_graph_dataset(num_graphs=500, nodes_per_graph=10):
    X_list = []
    A_list = []
    Y_list = []
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    for _ in range(num_graphs):
        is_fraud = np.random.choice([0, 1], p=[0.9, 0.1])
        
        if is_fraud == 0:
            features = np.random.normal(loc=0.5, scale=0.2, size=(nodes_per_graph, 4))
            # Sparse network representing standard user interactions
            adj = np.eye(nodes_per_graph)
            adj[0, 1] = 1.0; adj[1, 0] = 1.0
        else:
            features = np.random.normal(loc=0.8, scale=0.1, size=(nodes_per_graph, 4))
            # Dense network representing rapid wash trading / pump connections
            adj = np.ones((nodes_per_graph, nodes_per_graph)) / nodes_per_graph
            
        X_list.append(torch.tensor(features, dtype=torch.float32))
        A_list.append(torch.tensor(adj, dtype=torch.float32))
        Y_list.append(torch.tensor([[float(is_fraud)]], dtype=torch.float32))
        
    return X_list, A_list, Y_list

def train_gnn():
    logging.info("Synthesizing Relational Transaction Graph dataset...")
    X, A, Y = synthesize_graph_dataset()
    
    model = GNNFraudDetector(input_dim=4, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    logging.info("Initiating strict Deep Learning PyTorch Backpropagation execution.")
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(X)):
            optimizer.zero_grad()
            
            x_var = X[i]
            a_var = A[i]
            y_true = Y[i]
            
            # Predict only anchor node (index 0) mapping relationship bounds logically
            y_pred = model(x_var, a_var)[0].unsqueeze(0)
            
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(X):.4f}")
            
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Deep Learning PyTorch weights persisted natively to {MODEL_PATH}")

if __name__ == "__main__":
    train_gnn()
