# Active Learning for SmokeyNet

This implementation provides an active learning framework for the SmokeyNet fire detection model using Gaussian Process uncertainty estimation.

## Overview

The active learning system:
1. **Loads the trained SmokeyNet model** from `model.onnx`
2. **Uses Gaussian Process regression** to estimate uncertainty in predictions
3. **Performs uncertainty-based acquisition** to select the most informative samples
4. **Tracks performance metrics** across active learning iterations
5. **Visualizes results** with comprehensive plots

## Key Features

### Gaussian Process Uncertainty Estimation
- Uses RBF kernel with constant kernel for feature similarity
- Estimates prediction uncertainty for unlabeled samples
- Selects samples with highest uncertainty for annotation

### Active Learning Pipeline
- **Initial Pool**: Randomly selected samples to start training
- **Unlabeled Pool**: Remaining samples available for selection
- **Test Pool**: Held-out samples for performance evaluation
- **Acquisition Strategy**: Uncertainty-based selection

### Performance Tracking
- Accuracy, Precision, Recall, F1-Score
- Uncertainty scores of selected samples
- Number of samples selected per iteration

## File Structure

```
├── active_learning_smokeynet.py    # Main active learning script
├── requirements_active_learning.txt  # Dependencies
├── README_active_learning.md        # This file
├── src/
│   ├── inference/
│   │   ├── smokeynet.py            # SmokeyNet inference class
│   │   └── model.onnx              # Trained model
│   └── groundtruth/
│       └── results/
│           └── ground_truth_combined.json  # Ground truth labels
└── extracted/                       # Image data directories
    └── [firename]/
        └── [timestamp]_[offset].jpg
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements_active_learning.txt
```

2. Ensure the model file exists:
```bash
ls src/inference/model.onnx
```

## Usage

Run the active learning script:

```bash
python active_learning_smokeynet.py
```

### Configuration

You can modify the active learning parameters in the `main()` function:

```python
# Active learning parameters
initial_pool_size = 200      # Number of samples to start with
acquisition_batch_size = 50  # Samples to acquire per iteration
max_iterations = 8          # Maximum active learning iterations
```

### Output Files

The script generates:
- `active_learning_performance.png`: Performance plots
- `active_learning_results.pkl`: Detailed results data

## Active Learning Process

1. **Data Preparation**:
   - Load ground truth labels from JSON
   - Group images by firename for temporal consistency
   - Split into labeled, unlabeled, and test pools

2. **Feature Extraction**:
   - Use SmokeyNet to extract tile probabilities as features
   - Normalize features to fixed size

3. **Gaussian Process Training**:
   - Train GP model on labeled data
   - Use RBF kernel for feature similarity

4. **Uncertainty-Based Acquisition**:
   - Calculate uncertainty for unlabeled samples
   - Select samples with highest uncertainty
   - Move selected samples to labeled pool

5. **Performance Evaluation**:
   - Evaluate on test set using GP predictions
   - Track accuracy, precision, recall, F1-score

6. **Iteration**:
   - Repeat steps 3-5 for specified iterations
   - Plot performance metrics

## Performance Metrics

The system tracks:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Uncertainty**: Mean uncertainty of selected samples
- **Sample Count**: Number of samples selected per iteration

## Visualization

The script creates a comprehensive 6-panel plot showing:
1. Accuracy over iterations
2. Precision over iterations
3. Recall over iterations
4. F1-Score over iterations
5. Mean uncertainty of selected samples
6. Number of samples selected per iteration

## Customization

### Feature Extraction
Modify `get_image_features()` to use different features:
- Raw pixel values
- Pre-trained CNN features
- Custom feature extractors

### Acquisition Strategy
Modify `select_uncertain_samples()` for different strategies:
- Expected Model Change
- Query-by-Committee
- Information Gain

### Gaussian Process Kernel
Modify `train_gaussian_process()` to use different kernels:
- Matern kernel
- Polynomial kernel
- Custom kernels

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or use data streaming
2. **Slow Feature Extraction**: Use GPU acceleration or batch processing
3. **Poor Performance**: Adjust GP hyperparameters or feature extraction

### Performance Tips

1. **Use GPU**: Ensure PyTorch uses GPU if available
2. **Batch Processing**: Process images in batches for efficiency
3. **Caching**: Cache extracted features to avoid recomputation
4. **Parallel Processing**: Use multiprocessing for feature extraction

## Advanced Usage

### Custom Data Loading
```python
# Modify prepare_data_pools() for custom data sources
def prepare_data_pools(self):
    # Your custom data loading logic
    pass
```

### Custom Uncertainty Estimation
```python
# Implement different uncertainty measures
def get_uncertainty_scores(self, X):
    # Your custom uncertainty calculation
    return uncertainties
```

### Custom Acquisition Strategy
```python
# Implement different acquisition strategies
def select_samples(self, unlabeled_features, unlabeled_samples, n_samples):
    # Your custom selection strategy
    return selected_samples
```

## References

- SmokeyNet: A Deep Learning Framework for Wildland Fire Detection and Analysis
- Active Learning with Gaussian Processes
- Uncertainty Estimation in Deep Learning 