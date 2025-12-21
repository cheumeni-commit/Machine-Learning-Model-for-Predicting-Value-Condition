"""
Script d'entraînement du modèle avec intégration MLflow
Entraîne le modèle et enregistre tous les résultats dans MLflow
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Importer le gestionnaire MLflow
from training.mlflow_manager import MLflowManager, train_and_log_model
from read_write import load_data_train_test
from constants import c_CLASSES
from training.evaluation import evaluate_model
from training.models import get_models
from config.directories import directories as dirs
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data():
    """
    Load data train and test from files.
    
    Args:
        None
    Returns:
        X_train: DataFrame
        y_train: Series
        X_test: DataFrame
        y_test: Series
    """
    try:
        logger.info("Loading data train and test...")
        X_train, X_test, y_train, y_test = load_data_train_test()
        logger.info("Data train and test loaded. ✅")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        return None, None, None, None


def optimize_model_pipeline(GridMethod, 
                            X_train, 
                            y_train, 
                            model, 
                            param_grid, 
                            cv=5, 
                            scoring_fit='accuracy',
                            ):
    """
    Optimize model pipeline.
    
    Args:
        GridMethod: GridSearchCV or RandomizedSearchCV
        X_train: X_train data
        y_train: y_train data
        model: model
        param_grid: param_grid
        cv: cross-validation
        scoring_fit: scoring function
    Returns:
        fitted_model: fitted model
    """
    if GridMethod == "GridSearchCV":
        
        gs = GridSearchCV(
            estimator=model(),
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
        )
        fitted_model = gs.fit(X_train, y_train)
        
    else:
        cs = RandomizedSearchCV(
            estimator=model(),
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
        )
        fitted_model = cs.fit(X_train, y_train)
    
    return fitted_model


def train_optimized_model(X_train, y_train, X_test, y_test, mlflow_manager):
    """
    Entraîner un modèle optimisé avec GridSearchCV.
    
    Args:
        X_train, y_train, X_test, y_test: Données
        mlflow_manager: Gestionnaire MLflow
    """
    print("\n" + "="*50)
    print("ENTRAÎNEMENT DU MODÈLE OPTIMISÉ (GridSearchCV)")
    print("="*50)

    logger.info("Optimizing model pipeline...")

    models, param_models = get_models()

    metrics_train, metrics_test = [], []
    best_model = []
    name_model = []
    for model, param_model in zip(models, param_models):

        logger.info(f"Training model...")

        fitted_model = optimize_model_pipeline("GridSearchCV", X_train, y_train, model, param_model)

        logger.info("Model optimized.")

        best_model.append(fitted_model)
        name_model.append(f"optimized_{fitted_model.best_estimator_.__class__.__name__}")
        logger.info("Model trained.")

        
        train_metrics, test_metrics = (
             evaluate_model(fitted_model.best_estimator_, X=features[0], y=features[1], classes=c_CLASSES)
             for features in ((X_train, y_train), (X_test, y_test))
        )
        print(train_metrics)
        print(test_metrics)
        metrics_test.append(test_metrics)
        metrics_train.append(train_metrics)

        metrics_msg = "=" * 50 + " Metrics " + "=" * 50
        logger.info(metrics_msg)
        logger.info(fitted_model)
        logger.info(f"Train: {train_metrics.get('overall')}")
        logger.info(f"Test: {test_metrics.get('overall')}")
        logger.info("=" *len(metrics_msg))

     # choix du meilleur model en s'appuyant sur le f1_score
    f1_score = [res.get('overall').get('f1') for res in metrics_test]
    index_best_model = np.argmax(f1_score)
    best_model = best_model[index_best_model].best_estimator_

    # enregistrement du meilleur model
    model, metrics = train_and_log_model(
        X_train, y_train, X_test, y_test, best_model,
        mlflow_manager,
        run_name=f"best_{best_model.__class__.__name__}"
    )


    return best_model, metrics


def main(): 
    """Fonction principale."""
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE AVEC MLFLOW")
    print("="*60)
    
    # Initialiser MLflow
    mlflow_manager = MLflowManager(
        experiment_name="Valve Condition Predictor",
        tracking_uri=f"file:{dirs.raw_store_dir}"
    )
    
    # Charger les données
    try:
        X_train, X_test, y_train, y_test = load_data()
        logger.info("Data train and test loaded. ✅")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        print("Utilisation des données de test synthétiques...")
        
    
    # Entraîner le modèle optimisé
    optimized_model = train_optimized_model(X_train, y_train, X_test, y_test, mlflow_manager)
    
    # Obtenir la meilleure exécution
    print("\n" + "="*50)
    print("RÉSUMÉ DES EXÉCUTIONS")
    print("="*50)
    
    best_run = mlflow_manager.get_best_run(metric="test_accuracy")
    
    if best_run:
        print(f"\nMeilleure exécution: {best_run['run_id']}")
        print(f"Métriques:")
        for key, value in best_run['metrics'].items():
            print(f"  {key}: {value}")
    
    # Afficher toutes les exécutions
    print("\n" + "="*50)
    print("TOUTES LES EXÉCUTIONS")
    print("="*50)
    
    runs = mlflow_manager.get_experiment_runs()
    print(f"\nNombre d'exécutions: {len(runs)}")
    print("\nRésumé des exécutions:")
    
    for idx, run in runs.iterrows():
        print(f"\nExécution: {run['run_id']}")
        print(f"  Nom: {run['tags.mlflow.runName']}")
        print(f"  Status: {run['status']}")
        if 'metrics.test_accuracy' in run and pd.notna(run['metrics.test_accuracy']):
            print(f"  Accuracy: {run['metrics.test_accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*50)
    print("\nPour visualiser les résultats MLflow:")
    print("  mlflow ui --backend-store-uri file:./mlruns")
    print("\nPuis accédez à http://localhost:5000")


if __name__ == "__main__":
    main()
