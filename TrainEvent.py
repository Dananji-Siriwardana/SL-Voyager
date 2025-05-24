import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load data
data_file_path = 'Dataset.csv'
event_data = pd.read_csv(data_file_path)

# Feature engineering
X = event_data[['Event_Type', 'Guest_Count', 'Budget_Range', 'Special_Requirements']]
y = event_data['Venue_Name']

# Convert numerical format
X['Min_Budget'] = X['Budget_Range'].apply(lambda x: int(x.split()[1]))
X['Max_Budget'] = X['Budget_Range'].apply(lambda x: int(x.split()[1]))
X.drop('Budget_Range', axis=1, inplace=True)

# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
X['Guest_Count'] = imputer.fit_transform(X[['Guest_Count']])
X['Min_Budget'] = imputer.fit_transform(X[['Min_Budget']])
X['Max_Budget'] = imputer.fit_transform(X[['Max_Budget']])

# Transformer for categorical features
categorical_features = ['Event_Type', 'Special_Requirements']
numerical_features = ['Guest_Count', 'Min_Budget', 'Max_Budget']

# Model pipeline with scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Random forest model with class weights
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight=class_weight_dict))
])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Save the model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'event_recommendation_model.pkl')

# Evaluation
y_pred_proba = best_model.predict_proba(X_test)
top3_pred_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
top3_pred = best_model.named_steps['classifier'].classes_[top3_pred_indices]
top3_accuracy = np.mean([1 if y_test.iloc[i] in top3_pred[i] else 0 for i in range(len(y_test))])

print("\nClassification Report (Top-1 Prediction):")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\nTop-3 Accuracy:", top3_accuracy)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"\n------------Model saved------------")