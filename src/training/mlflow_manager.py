"""
Module MLflow pour le tracking, la sauvegarde et la gestion des modèles
Permet de suivre les expériences, les métriques et les versions des modèles
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import logging

from src.config.directories import directories as dirs

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowManager:
    """Gestionnaire MLflow pour le tracking et la sauvegarde des modèles."""
    
    def __init__(self, experiment_name: str = "Valve Condition Predictor", 
                 tracking_uri: str = f"file:{dirs.raw_store_dir}"):
        """
        Initialiser le gestionnaire MLflow.
        
        Args:
            experiment_name: Nom de l'expérience
            tracking_uri: URI du serveur MLflow (local par défaut)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Configurer MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Créer ou récupérer l'expérience
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id if experiment else None
        
        if self.experiment_id:
            mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow configuré - Expérience: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """
        Démarrer une nouvelle exécution MLflow.
        
        Args:
            run_name: Nom de l'exécution
            tags: Tags pour l'exécution (dict)
            
        Returns:
            ID de l'exécution
        """
        run = mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        logger.info(f"Exécution démarrée: {run.info.run_id}")
        return run.info.run_id
    
    def end_run(self):
        """Terminer l'exécution courante."""
        mlflow.end_run()
        logger.info("Exécution terminée")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Enregistrer les paramètres.
        
        Args:
            params: Dictionnaire des paramètres
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Paramètres enregistrés: {len(params)} paramètres")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Enregistrer les métriques.
        
        Args:
            metrics: Dictionnaire des métriques
            step: Étape (optionnel)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Métriques enregistrées: {len(metrics)} métriques")
    
    def log_model(self, model, model_name: str = "valve_condition_model", 
                  artifact_path: str = "models"):
        """
        Enregistrer le modèle.
        
        Args:
            model: Modèle scikit-learn
            model_name: Nom du modèle
            artifact_path: Chemin d'artefact
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=model_name
        )
        logger.info(f"Modèle enregistré: {model_name}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        Enregistrer un artefact (fichier).
        
        Args:
            local_path: Chemin local du fichier
            artifact_path: Chemin d'artefact dans MLflow
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info(f"Artefact enregistré: {local_path}")
    
    def log_dataframe(self, df: pd.DataFrame, name: str = "data"):
        """
        Enregistrer un DataFrame.
        
        Args:
            df: DataFrame à enregistrer
            name: Nom du fichier
        """
        csv_path = dirs.raw_store_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        os.remove(csv_path)
        logger.info(f"DataFrame enregistré: {name}")
    
    def log_feature_importance(self, feature_names: list, importances: np.ndarray):
        """
        Enregistrer l'importance des features.
        
        Args:
            feature_names: Noms des features
            importances: Valeurs d'importance
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        self.log_dataframe(importance_df, "feature_importance")
        logger.info(f"Importance des features enregistrée")
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Enregistrer la matrice de confusion.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=['Non-Optimal', 'Optimal'],
            columns=['Non-Optimal', 'Optimal']
        )
        
        self.log_dataframe(cm_df, "confusion_matrix")
        logger.info(f"Matrice de confusion enregistrée")
    
    def log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Enregistrer le rapport de classification.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
        """
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        self.log_dataframe(report_df, "classification_report")
        logger.info(f"Rapport de classification enregistré")
    
    def get_best_run(self, metric: str = "test_accuracy") -> Optional[Dict[str, Any]]:
        """
        Obtenir la meilleure exécution selon une métrique.
        
        Args:
            metric: Nom de la métrique
            
        Returns:
            Dictionnaire avec les informations de la meilleure exécution
        """
        experiment = mlflow.get_experiment(self.experiment_id)
        
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0].to_dict()

        return {
            'run_id': best_run['run_id'],
            'metrics': {col: best_run[col] for col in best_run.keys() if col.startswith('metrics.')},
            'params': {col: best_run[col] for col in best_run.keys() if col.startswith('params.')},
            'timestamp': best_run['start_time']
        }
    
    def load_model(self, run_id: str, artifact_path: str = "models"):
        """
        Charger un modèle depuis une exécution.
        
        Args:
            run_id: ID de l'exécution
            artifact_path: Chemin d'artefact
            
        Returns:
            Modèle chargé
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modèle chargé: {run_id}")
        return model
    
    def get_experiment_runs(self) -> pd.DataFrame:
        """
        Obtenir toutes les exécutions de l'expérience.
        
        Returns:
            DataFrame avec les exécutions
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def compare_runs(self, run_ids: list) -> pd.DataFrame:
        """
        Comparer plusieurs exécutions.
        
        Args:
            run_ids: Liste des IDs d'exécution
            
        Returns:
            DataFrame de comparaison
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        comparison = runs[runs['run_id'].isin(run_ids)]
        return comparison
    
    def export_run(self, run_id: str, export_path: str):
        """
        Exporter une exécution (modèle et artefacts).
        
        Args:
            run_id: ID de l'exécution
            export_path: Chemin d'export
        """
        os.makedirs(export_path, exist_ok=True)
        
        # Copier les artefacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        for artifact in artifacts:
            local_path = client.download_artifacts(run_id, artifact.path, export_path)
            logger.info(f"Artefact exporté: {local_path}")
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Obtenir les informations d'une exécution.
        
        Args:
            run_id: ID de l'exécution
            
        Returns:
            Dictionnaire avec les informations
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }


def train_and_log_model(X_train, y_train, X_test, y_test, model, 
                       mlflow_manager: MLflowManager, run_name: str = None):
    """
    Entraîner un modèle et enregistrer les résultats dans MLflow.
    
    Args:
        X_train: Données d'entraînement
        y_train: Labels d'entraînement
        X_test: Données de test
        y_test: Labels de test
        model: Modèle à entraîner
        mlflow_manager: Gestionnaire MLflow
        run_name: Nom de l'exécution
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Démarrer une exécution MLflow
    mlflow_manager.start_run(
        run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={
            'model_type': type(model).__name__,
            'dataset': 'valve_condition',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    try:
        # Entraîner le modèle
        model.fit(X_train, y_train)
        
        # Faire des prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculer les métriques
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted')
        }
        
        # Enregistrer les paramètres
        params = model.get_params()
        mlflow_manager.log_params(params)
        
        # Enregistrer les métriques
        mlflow_manager.log_metrics(metrics)
        
        # Enregistrer le modèle
        mlflow_manager.log_model(model)
        
        # Enregistrer l'importance des features
        if hasattr(model, 'feature_importances_'):
            mlflow_manager.log_feature_importance(
                X_test.columns.tolist(),
                model.feature_importances_
            )
        
        # Enregistrer la matrice de confusion
        mlflow_manager.log_confusion_matrix(y_test, y_pred_test)
        
        # Enregistrer le rapport de classification
        mlflow_manager.log_classification_report(y_test, y_pred_test)
        
        logger.info(f"Modèle entraîné et enregistré avec succès")
        logger.info(f"Métriques: {metrics}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        raise
    finally:
        mlflow_manager.end_run()


def load_best_model(mlflow_manager: MLflowManager, metric: str = "test_accuracy"):
    """
    Charger le meilleur modèle selon une métrique.
    
    Args:
        mlflow_manager: Gestionnaire MLflow
        metric: Métrique à utiliser
        
    Returns:
        Modèle chargé
    """
    best_run = mlflow_manager.get_best_run(metric)
    
    if not best_run:
        logger.warning("Aucune exécution trouvée")
        return None
    
    model = mlflow_manager.load_model(best_run['run_id'])
    logger.info(f"Meilleur modèle chargé (run_id: {best_run['run_id']})")
    
    return model
