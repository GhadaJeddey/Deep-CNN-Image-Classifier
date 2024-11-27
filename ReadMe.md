1. Understanding LiDAR and Point Cloud Data \\
What is LiDAR?
LiDAR (Light Detection and Ranging) employs laser light to measure distances, generating 3D
representations of environments. The project processes point cloud data from LiDAR sensors
to classify objects.
Key Concepts Learned:
• Point Clouds: Sets of 3D points, represented by coordinates (x, y, z).
• Spherical vs. Cartesian Coordinates: LiDAR typically uses spherical coordinates,
while the PointNet model requires Cartesian coordinates. Transformation logic needs
to be implemented.
• Data Augmentation: Techniques like random rotation, jittering, and normalization
enhance model robustness.
• File Formats: Worked with .npz files to store and load point cloud data.
2. Dataset Preparation
Custom Dataset Class
The LiDARData class was developed to handle loading, preprocessing, and augmenting point
cloud data.
Key Features:
• Partitioning Data: Split into training and testing datasets.
• Augmentation Techniques:
o Random rotation around the Z-axis.
o Adding noise (jitter) to points.
• Normalization:
o Centering the point cloud by subtracting its centroid.
o Scaling to fit within a unit sphere by dividing by the maximum distance.
Lessons Learned:
• Leveraged PyTorch's torch.utils.data.Dataset to create a custom dataset.
• Dynamically randomized and augmented data during loading.
3. Deep Learning with PointNet
Model Architecture
PointNet is designed to process unordered point cloud data. It uses shared MLPs (Multi-Layer
Perceptrons) to extract features and applies max pooling for global feature representation.
Implementation Highlights:
• Shared MLP Layers:
o Used Conv1d layers for point-wise feature extraction.
o Integrated BatchNorm1d for stable training.
• Global Feature Extraction:
o Max pooling was applied to capture the most significant features.
• Classification Layers:
o Fully connected layers with dropout for regularization.
• Softmax Output:
o Provided class probabilities.
Key Learnings:
• Permutation Invariance: Ensured that the model's results were unaffected by the
order of points in the input data.
• Used ReLU activations and Dropout to improve generalization.
References:
• PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
4. Training the Model
Process
The model was trained using augmented point cloud data with:
• Loss Function: Cross-entropy loss for classification.
• Optimizer: Adam optimizer for efficient parameter updates.
• Validation: Used validation accuracy to select the best model.
Insights Gained:
• Split datasets using PyTorch's random_split.
• Tracked and saved the best model during training.
• Evaluated using metrics like accuracy (sklearn.metrics.accuracy_score).
5. Model Quantization
What is Quantization?
Quantization reduces model size and inference latency by converting weights and activations
from floating-point to lower precision (e.g., 8-bit integers).
Process:
• Applied PyTorch’s quantize_dynamic to linear and convolutional layers.
• Exported and saved the quantized model for deployment.
Benefits:
• Faster inference and reduced memory usage.
• Minimal accuracy loss when applied correctly.
6. Real-Time Inference and Streaming
LiDARStreamer
Developed the LiDARStreamer class to:
• Fetch live point cloud data from a LiDAR device.
• Normalize and preprocess data for inference.
• Use the trained model for real-time object classification.
Serial Communication
• Used Python’s serial module to send classification results to external devices over
UART.
• Implemented cross-platform serial communication.
