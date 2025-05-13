import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
# from sklearn.preprocessing import label_binarize # Not used in current example
import numpy as np
import pandas as pd # For DataFrame creation

# --- Import Scikit-learn models and tuning tools ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV # For hyperparameter tuning

from kusa.client import SecureDatasetClient
# Conditional TF/PyTorch imports (kept for completeness of factory, but not used if framework is sklearn)
try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = None; nn = None; TensorDataset = None; DataLoader = None


# 🔧 Framework flag
TRAINING_FRAMEWORK = "sklearn" # Focus on sklearn for this enhancement
TARGET_COLUMN = "sentiment" # Or your actual target: "Category", "RainTomorrow"

# --- CHOOSE YOUR SKLEARN MODEL HERE ---
SELECTED_SKLEARN_MODEL = LogisticRegression # Example: Start with a robust baseline
# SELECTED_SKLEARN_MODEL = RandomForestClassifier
# SELECTED_SKLEARN_MODEL = SVC 
# SELECTED_SKLEARN_MODEL = GradientBoostingClassifier
# SELECTED_SKLEARN_MODEL = MultinomialNB # Good for TF-IDF if no scaling of numerics is done

load_dotenv(override=True)


# In main.py

def train_model_factory(framework="sklearn", model_class=None, fixed_params=None, param_grid=None):
    fixed_params = fixed_params or {}
    if framework == "sklearn":
        def train_model(X, y, **runtime_params): # Renamed params to runtime_params for clarity
            if param_grid: # If a param_grid is provided, use GridSearchCV
                print(f"  Hyperparameter tuning {model_class.__name__} with GridSearchCV...")
                base_estimator_init_params = {k:v for k,v in fixed_params.items() if k in signature(model_class.__init__).parameters}
                grid_search = GridSearchCV(model_class(**base_estimator_init_params), param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
                grid_search.fit(X, y)
                print(f"  Best parameters found for {model_class.__name__}: {grid_search.best_params_}")
                return grid_search.best_estimator_
            else:
                # Standard training with fixed/passed hyperparameters
                sig = signature(model_class.__init__)
                accepted_init_params = set(sig.parameters.keys()) # Correct variable name
                
                # Prioritize runtime_params over fixed_params if there's an overlap
                all_params = {**fixed_params, **runtime_params} 
                
                # Use the correct variable 'accepted_init_params' here
                valid_init_params = {k: v for k, v in all_params.items() if k in accepted_init_params} # <--- CORRECTED
                
                print(f"  Initializing {model_class.__name__} with params: {valid_init_params}")
                model = model_class(**valid_init_params)
                print(f"  Fitting {model_class.__name__}...")
                model.fit(X, y)
                return model
        return train_model
    # ... (TensorFlow and PyTorch factory parts remain the same) ...
    elif framework == "tensorflow":
        # ... your TF factory ...
        pass
    elif framework == "pytorch":
        # ... your PyTorch factory ...
        pass
    else:
        raise ValueError(f"Unsupported framework for this factory example: {framework}")


def plot_confusion_matrix(y_true, y_pred, model_name, title="Confusion Matrix"): # Added model_name
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - {title}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show(block=False)

def plot_precision_recall_curve(y_true, y_proba, model_name, title="Precision-Recall Curve"): # Added model_name
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    plt.figure(); plt.plot(recall, precision, label=f"AP={avg_precision:.2f}"); plt.xlabel("Recall")
    plt.ylabel("Precision"); plt.title(f"{model_name} - {title}"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)

def plot_threshold_analysis(y_true, y_proba, model_name, title="Threshold Analysis"): # Added model_name
    thresholds = np.linspace(0.01, 0.99, 100); precisions = []; recalls = []; f1s = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if np.sum(preds) > 0 and np.sum(y_true) > 0 : 
            p_val, r_val, _ = precision_recall_curve(y_true, preds, pos_label=1)
            precisions.append(p_val[1] if len(p_val) > 1 else 0.0) 
            recalls.append(r_val[1] if len(r_val) > 1 else 0.0)
        else: 
            precisions.append(0.0); recalls.append(0.0)
        f1s.append(f1_score(y_true, preds, zero_division=0, pos_label=1)) # Specify pos_label for binary
    plt.figure(figsize=(8, 5)); plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green"); plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(x=0.5, linestyle='--', color='gray', label="Thresh=0.5"); plt.xlabel("Threshold")
    plt.ylabel("Score"); plt.title(f"{model_name} - {title}"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)


def main():
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")

    print(f"\n--- Starting Workflow for Kusa SDK with Scikit-learn ---")
    print(f"--- Using Model: {SELECTED_SKLEARN_MODEL.__name__} ---")

    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    
    print("🚀 Initializing SDK...")
    initialization = client.initialize()
    if not initialization or "metadata" not in initialization:
        print("❌ SDK Initialization failed."); return
    print(f"SDK Initialized. Total data rows: {initialization['metadata']['totalDataRows']}")
    # client.preview() was in your code, ensure it exists or use initialization['preview']
    # print("Data Preview:\n", initialization.get('preview', pd.DataFrame()))


    print(" Fetching entire dataset...")
    client.fetch_and_decrypt_batch(batch_size=10000) # SDK fetches all data

    print("⚙️ Configuring preprocessing...")
    # --- Preprocessing Configuration ---
    # This is a critical step for good model performance
    preprocessing_config = { 
        "tokenizer": "nltk",      # "spacy" can be better but slower
        "stopwords": True,
        "lowercase": True,
        "remove_punctuation": True,
        "lemmatize": False,       # Set to True with "spacy" for potential improvement
        "reduction": "tfidf",     # Good default for text.
                                  # Options: "none", "tfidf", "pca", "tfidf_pca"
        "tfidf_max_features": 2000, # More features for TF-IDF can help
        "n_components": 50,         # If using PCA, more components often better than 2
        "target_column": TARGET_COLUMN,
        "target_encoding": "auto"
    }
    # Note: If your dataset has significant numeric features and you use "tfidf" or "none"
    # for reduction, models like LogisticRegression, SVC, KNN benefit from scaling these numeric features.
    # Your PreprocessingManager would need an option for "scale_numeric_features": True/False
    # or you'd rely on "pca" which includes scaling.
    # For MultinomialNB, ensure features are non-negative (TF-IDF is fine).

    client.configure_preprocessing(preprocessing_config)
    client.run_preprocessing()



    print(f"🎯 Building training function for {SELECTED_SKLEARN_MODEL.__name__}...")
    
    # --- Hyperparameter Setup & Optional GridSearchCV ---
    # Set to True to use GridSearchCV for the selected model
    USE_GRID_SEARCH = False # CHANGE THIS TO True TO ENABLE GRID SEARCH FOR THE SELECTED MODEL

    param_grid_for_tuning = None
    # Define fixed_params (will be used if not tuning that param, or as part of base estimator for GridSearchCV)
    # and param_grid (for GridSearchCV to search over)
    fixed_hyperparams = {"random_state": 42} # Common params

    if SELECTED_SKLEARN_MODEL == RandomForestClassifier:
        fixed_hyperparams.update({"class_weight": "balanced"})
        if USE_GRID_SEARCH:
            param_grid_for_tuning = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10]
            }
        else:
            fixed_hyperparams.update({"n_estimators": 100, "max_depth": 20})
    
    elif SELECTED_SKLEARN_MODEL == LogisticRegression:
        fixed_hyperparams.update({"solver": "liblinear", "class_weight": "balanced"})
        if USE_GRID_SEARCH:
            param_grid_for_tuning = {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"]
            }
        else:
            fixed_hyperparams.update({"C": 1.0})

    elif SELECTED_SKLEARN_MODEL == SVC:
        fixed_hyperparams.update({"class_weight": "balanced", "probability": True}) # probability=True for predict_proba
        if USE_GRID_SEARCH:
            param_grid_for_tuning = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"] # if kernel is rbf
            }
        else:
            fixed_hyperparams.update({"C": 1.0, "kernel": "rbf"})
            
    elif SELECTED_SKLEARN_MODEL == GradientBoostingClassifier:
        # GBC doesn't have class_weight directly, handle imbalance via sample_weight in fit or other methods.
        # For simplicity, not adding sample_weight to factory here.
        if USE_GRID_SEARCH:
             param_grid_for_tuning = {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5]
            }
        else:
            fixed_hyperparams.update({"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3})
            
    elif SELECTED_SKLEARN_MODEL == MultinomialNB:
        if USE_GRID_SEARCH:
            param_grid_for_tuning = {"alpha": [0.1, 0.5, 1.0, 2.0]}
        else:
            fixed_hyperparams.update({"alpha": 1.0})
    # Add more model-specific hyperparameter configurations and grids as needed

    # The train_model_factory now takes the param_grid
    train_model_func = train_model_factory(
        framework="sklearn", 
        model_class=SELECTED_SKLEARN_MODEL,
        fixed_params=fixed_hyperparams, # Params used if not in grid, or as base for GridSearch
        param_grid=param_grid_for_tuning if USE_GRID_SEARCH else None
    )
           
    print("🚀 Training model...")
    # If using GridSearchCV, hyperparams passed to client.train might be minimal or empty,
    # as GridSearchCV handles the search. The factory passes them through.
    # For standard training, fixed_hyperparams are used.
    client.train(
         user_train_func=train_model_func, 
         hyperparams={}, # Pass empty if GridSearchCV is handling it, or pass overrides
         target_column=TARGET_COLUMN,
         task_type="classification", 
         framework=TRAINING_FRAMEWORK
    )

    print("📈 Evaluating model...")
    results = client.evaluate()
    if results:
        print("\n✅ Evaluation Accuracy:", results.get("accuracy", "N/A"))
        print("📊 Classification Report:\n", results.get("report", "N/A"))

    # --- Visualizations ---
    y_true_val = client._SecureDatasetClient__y_val
    X_val_processed = client._SecureDatasetClient__X_val
    trained_model_obj = client._SecureDatasetClient__trained_model
    model_name_for_plot = SELECTED_SKLEARN_MODEL.__name__

    if y_true_val is not None and X_val_processed is not None and not X_val_processed.empty and trained_model_obj is not None:
        print(f"📉 Visualizing performance for {model_name_for_plot}...")
        y_pred_classes = client.predict(client._SecureDatasetClient__X_val)
        plot_confusion_matrix(y_true_val, y_pred_classes, model_name_for_plot)

        if hasattr(trained_model_obj, "predict_proba"):
            y_pred_proba = trained_model_obj.predict_proba(X_val_processed)[:, 1]
            plot_precision_recall_curve(y_true_val, y_pred_proba, model_name_for_plot)
            plot_threshold_analysis(y_true_val, y_pred_proba, model_name_for_plot)
        else:
            print(f"   Note: {model_name_for_plot} does not have predict_proba method. Skipping PR and Threshold plots.")
    else:
        print("   Skipping visualizations: validation data or model unavailable.")

    # --- Saving ---
    print("💾 Saving trained model...")
    save_filename = f"secure_model_{model_name_for_plot.lower()}.ksmodel"
    client.save_model(save_filename)

    print("\n✅ Done! 🎉")

if __name__ == "__main__":
    main()
    if plt.get_fignums():
        print("\nDisplaying plots... Close plot windows to exit script.")
        plt.show(block=True)