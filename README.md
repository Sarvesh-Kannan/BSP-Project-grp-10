## Drowsiness Detection using Machine Learning

### Overview
This project is adapted from research on traditional ML methods for sleep scoring, repurposed here for drowsiness detection. We utilize EEG, EOG, and EMG signals to identify drowsiness episodes in real-time, leveraging the benefits of interpretable and efficient traditional ML models, which offer competitive performance to deep learning with less complexity and higher adaptability to clinical use.

### Objective
The goal is to provide an ML-based pipeline for drowsiness detection that can achieve high accuracy while maintaining interpretability and reducing computational demands, thus enabling broader adoption in healthcare or in-vehicle systems.

### Key Features
- **Signal Processing**: Preprocess EEG, EOG, and EMG signals to remove noise and retain relevant frequency bands.
- **Feature Extraction**: Generate multi-domain and multi-resolution features from time and frequency domains to summarize each signal epoch.
- **Models**: Apply interpretable ML models such as logistic regression and gradient boosting to detect drowsiness states.
- **Temporal Context**: Integrate feature extraction from surrounding epochs for improved temporal consistency in drowsiness detection.

### Dataset
The project can be adapted to drowsiness datasets with similar polysomnography signals. Preprocess the data to focus on drowsiness events rather than sleep stages, which will serve as the primary labels.

### Pipeline
1. **Preprocessing**
   - EEG and EOG signals: Bandpass-filtered to 0.4–30 Hz.
   - EMG signal: Bandpass-filtered to 0.5–10 Hz.
   - Resampling to 100 Hz.
  
2. **Feature Extraction**
   - Time-domain and frequency-domain features extracted using `tsfresh` and `YASA` libraries.
   - Multi-resolution windows of 30s, 60s, and 90s with temporal context from neighboring epochs.
  
3. **Modeling**
   - **Logistic Regression**: A linear model that allows easy interpretation and handles high-dimensional feature vectors.
   - **Gradient Boosted Trees**: Non-linear model (CatBoost) for higher accuracy without complex hyperparameter tuning.
  
4. **Evaluation**
   - Train and evaluate using k-fold cross-validation.
   - Metrics: Accuracy, macro F1-score, and Cohen’s kappa to assess model performance and inter-rater reliability.
  
### Usage
1. Clone this repository.
2. Ensure required libraries are installed (e.g., `tsfresh`, `YASA`, `CatBoost`).
3. Preprocess your dataset following the pipeline above.
4. Run the model training script.
5. Evaluate using the provided cross-validation setup.

### Results
Expected performance metrics will vary by dataset, but the model aims to achieve competitive results compared to deep learning models in terms of accuracy, with enhanced interpretability and reduced computational requirements.

### Future Work
This approach may be expanded to real-time applications in driver drowsiness monitoring or other drowsiness detection systems in healthcare contexts.

### References
- **Source Paper**: [Do not sleep on traditional machine learning: Simple and interpretable techniques are competitive to deep learning for sleep scoring](https://doi.org/10.1016/j.bspc.2022.104429)

