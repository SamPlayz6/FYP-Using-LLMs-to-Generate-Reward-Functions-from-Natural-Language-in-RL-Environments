import torch
apiKey = "Insert API-KEY"
modelName = "claude-3-5-sonnet-20240620"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")