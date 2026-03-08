# Airline Passenger Referral Predictor

A machine learning model to predict whether airline passengers would recommend the airline based on their reviews and flight experience ratings.

## Project Description
This project analyzes airline passenger data from 2006-2019 to predict the likelihood of passengers recommending the airline to others. We implemented and compared multiple machine learning models including:

- Logistic Regression (92% accuracy)
- Decision Tree (92% accuracy after hyperparameter tuning)
- Random Forest (93% accuracy)
- K-Nearest Neighbors (93% accuracy) 
- Support Vector Machine (94% accuracy)
- Neural Network (94% accuracy)

The final deployed solution uses an ensemble approach combining predictions from all models through majority voting.

## Installation
1. Clone this repository:
```cmd
git clone [repository_url]
```

2. Navigate to project directory:
```cmd
cd "Referral Predictor"
```

3. Install requirements:
```cmd
pip install -r requirements.txt
```

## Usage

### Command Line:
1. First install required packages:
```cmd
pip install -r requirements.txt
```

2. Run the Streamlit prediction app:
```cmd
streamlit run app.py
```
![Screenshot 2025-03-24 195346](https://github.com/user-attachments/assets/442e59ef-e8d0-4f3c-9758-aae39f519acd)

The app will launch in your default browser at http://localhost:8501

### Web Interface:
- Use sliders to rate your flight experience (1-5)
- Select your traveller type and cabin class
- Click "Predict Recommendation" to see results

## Data
The dataset `data_airline_reviews.xlsx` contains 131,895 reviews from 2006-2019 with these key features:

### Rating Features (1-5 scale):
- Seat comfort (avg: 2.96)
- Cabin service (avg: 3.20)
- Food & beverage (avg: 2.93)
- Entertainment (avg: 2.89)
- Ground service (avg: 2.68)
- Value for money (avg: 2.95)

### Categorical Features:
- Traveller type (37% Solo Leisure, 28% Couple Leisure)
- Cabin class (77% Economy, 19% Business)

### Key Insights:
- Economy class had the lowest satisfaction ratings
- Cabin service and value for money were most predictive of recommendations
- 48% of passengers gave overall ratings ≥7 (likely to recommend)

## Model Details

### Neural Network Architecture:
- Input layer: 128 neurons (ReLU activation)
- Hidden layer: 64 neurons (ReLU activation) 
- Output layer: 1 neuron (Sigmoid activation)
- Optimizer: Adam (learning rate=0.01)
- Loss: Binary Crossentropy
- Early stopping with patience=15

### Machine Learning Models:
- Logistic Regression (C=1.0, penalty='l2')
- Decision Tree (max_depth=7, min_samples_split=5)
- Random Forest (n_estimators=30, max_depth=7)
- K-Nearest Neighbors (default parameters)
- Support Vector Machine (default parameters)

### Preprocessing:
- One-hot encoding for:
  - Traveller type (4 categories)
  - Cabin class (4 categories)
- Label encoding for target (recommended: yes/no)
- Feature scaling (StandardScaler)



## Results

### Model Performance Comparison:
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.92     | 0.91      | 0.93   | 0.92     |
| Decision Tree        | 0.92     | 0.90      | 0.94   | 0.92     |
| Random Forest        | 0.93     | 0.92      | 0.94   | 0.93     |
| K-Nearest Neighbors  | 0.93     | 0.92      | 0.94   | 0.93     |
| Support Vector Mach. | 0.94     | 0.93      | 0.95   | 0.94     |
| Neural Network       | 0.94     | 0.93      | 0.95   | 0.94     |

### Key Findings:
- SVM and Neural Network performed best with 94% accuracy
- All models achieved >90% accuracy after hyperparameter tuning
- Most important features for prediction:
  1. Value for money
  2. Cabin service
  3. Seat comfort
  4. Food & beverage quality

For detailed metrics and visualizations, see `experiments.ipynb`.
