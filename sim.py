import torch
from SwinUnet_3D import SwinUnet3D

# Set the hyperparameters for the SwinUnet3D model
hidden_dim = 96
layers = (2, 2, 4, 2)
heads = (3, 6, 9, 12)
num_classes = 2
window_size = 7
downscaling_factors = (4, 2, 2, 2)
relative_pos_embedding = True
dropout = 0.0

# Create an instance of the SwinUnet3D class
model = SwinUnet3D(hidden_dim=hidden_dim,
                   layers=layers,
                   heads=heads,
                   num_classes=num_classes,
                   window_size=window_size,
                   downscaling_factors=downscaling_factors,
                   relative_pos_embedding=relative_pos_embedding,
                   dropout=dropout)

# Load some example data
data = torch.randn(1, 1, 224, 224, 224)

# Make a prediction using the model
prediction = model(data)

# The output is a tensor representing the predicted segmentation mask for the input data
print(prediction.shape)
