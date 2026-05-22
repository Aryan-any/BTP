import os
import torch
import torch.nn as nn
from typing import List
from src.logger import logger

MODEL_PATH = "models/gnn_weights.pth"
_gnn_model = None

class GNNFraudDetector(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(GNNFraudDetector, self).__init__()
        # Standard structural fully-connected map 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.Linear(hidden_dim, 1)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.fc1(features))
        
        # Message Passing (Neighborhood aggregations)
        aggregated = torch.matmul(adj, hidden)
        
        out = self.fc2(aggregated)
        return self.sigmoid(out)

def _load_gnn():
    global _gnn_model
    if _gnn_model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error("Missing persistent Deep Learning GNN weights. Run 'scripts/train_gnn.py'")
                return False
            
            _gnn_model = GNNFraudDetector()
            _gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            _gnn_model.eval() # Set to inference
            logger.info("Deep Learning (PyTorch GNN) relational engine safely loaded natively.")
            return True
        except Exception as e:
            logger.error(f"GNN Inference Load Failure: {e}")
            return False
    return True

def predict_onchain_gnn(features: List[float], interactions: int = 1) -> float:
    if not _load_gnn():
        return 0.0

    try:
        # Dynamic adjacency formulation
        nodes = max(interactions, 1)
        # Create synthetic subgraph representing the transaction tree explicitly mapping
        f_tensor = torch.tensor([features] * nodes, dtype=torch.float32)
        
        # Complete graph connection (Worst-case relational map)
        adj_matrix = torch.ones((nodes, nodes), dtype=torch.float32) / nodes
        
        with torch.no_grad():
            output = _gnn_model(f_tensor, adj_matrix)
            return float(output[0].item())
    except Exception as e:
        logger.error(f"GNN Prediction Failure: {e}")
        return 0.0
