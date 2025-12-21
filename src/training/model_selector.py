"""
Module pour sélectionner et charger le meilleur modèle depuis le stockage MLflow
Permet de comparer les modèles basés sur leurs métriques et de charger le meilleur
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import yaml

from src.config.directories import directories as dirs

logger = logging.getLogger(__name__)


class ModelSelector:
    """Sélecteur de modèle pour choisir le meilleur modèle depuis le stockage MLflow."""
    
    def __init__(self, artifacts_dir: Path = None):
        """
        Initialiser le sélecteur de modèle.
        
        Args:
            artifacts_dir: Chemin vers le répertoire des artefacts MLflow
        """
        self.artifacts_dir = artifacts_dir or dirs.raw_store_dir
        self.tracking_uri = f"file:{self.artifacts_dir}"
        mlflow.set_tracking_uri(self.tracking_uri)
    
    def get_all_runs(self) -> List[Dict[str, Any]]:
        """
        Obtenir toutes les exécutions disponibles dans le stockage.
        
        Returns:
            Liste de dictionnaires contenant les informations de chaque exécution
        """
        runs_info = []
        
        # Parcourir le répertoire des artefacts
        experiments_dir = self.artifacts_dir
        
        if not experiments_dir.exists():
            logger.warning(f"Le répertoire {experiments_dir} n'existe pas")
            return runs_info
        
        # Chercher les répertoires d'expériences (IDs numériques)
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.isdigit():
                continue
            
            # Parcourir les runs dans chaque expérience
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                run_info = self._extract_run_info(run_dir)
                if run_info:
                    runs_info.append(run_info)
        
        return runs_info
    
    def _extract_run_info(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Extraire les informations d'une exécution depuis son répertoire.
        
        Args:
            run_dir: Répertoire de l'exécution
            
        Returns:
            Dictionnaire avec les informations de l'exécution ou None
        """
        try:
            run_id = run_dir.name
            
            # Lire les métriques
            metrics = {}
            metrics_dir = run_dir / "metrics"
            if metrics_dir.exists():
                for metric_file in metrics_dir.iterdir():
                    if metric_file.is_file():
                        try:
                            with open(metric_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    # Format MLflow: timestamp value step
                                    parts = lines[-1].strip().split()
                                    if len(parts) >= 2:
                                        value = float(parts[1])
                                        metrics[metric_file.name] = value
                        except Exception as e:
                            logger.warning(f"Erreur lors de la lecture de {metric_file}: {e}")
            
            # Lire les paramètres
            params = {}
            params_dir = run_dir / "params"
            if params_dir.exists():
                for param_file in params_dir.iterdir():
                    if param_file.is_file():
                        try:
                            with open(param_file, 'r') as f:
                                params[param_file.name] = f.read().strip()
                        except Exception as e:
                            logger.warning(f"Erreur lors de la lecture de {param_file}: {e}")
            
            # Lire les tags
            tags = {}
            tags_dir = run_dir / "tags"
            if tags_dir.exists():
                for tag_file in tags_dir.iterdir():
                    if tag_file.is_file():
                        try:
                            with open(tag_file, 'r') as f:
                                tags[tag_file.name] = f.read().strip()
                        except Exception as e:
                            logger.warning(f"Erreur lors de la lecture de {tag_file}: {e}")
            
            # Vérifier si un modèle existe
            model_path = run_dir / "artifacts" / "models"
            has_model = model_path.exists() and (model_path / "model.pkl").exists()
            
            return {
                'run_id': run_id,
                'metrics': metrics,
                'params': params,
                'tags': tags,
                'has_model': has_model,
                'run_dir': run_dir
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des informations de {run_dir}: {e}")
            return None
    
    def get_best_run(self, metric: str = "test_accuracy", 
                     higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """
        Obtenir la meilleure exécution selon une métrique.
        
        Args:
            metric: Nom de la métrique à utiliser (ex: "test_accuracy", "test_f1")
            higher_is_better: True si une valeur plus élevée est meilleure (False pour loss, etc.)
            
        Returns:
            Dictionnaire avec les informations de la meilleure exécution ou None
        """
        runs = self.get_all_runs()
        
        if not runs:
            logger.warning("Aucune exécution trouvée")
            return None
        
        # Filtrer les runs qui ont la métrique et un modèle
        valid_runs = [
            run for run in runs 
            if run['has_model'] and metric in run['metrics']
        ]
        
        if not valid_runs:
            logger.warning(f"Aucune exécution avec la métrique '{metric}' et un modèle trouvée")
            return None
        
        # Trier par métrique
        if higher_is_better:
            best_run = max(valid_runs, key=lambda x: x['metrics'][metric])
        else:
            best_run = min(valid_runs, key=lambda x: x['metrics'][metric])
        
        logger.info(f"Meilleure exécution trouvée: {best_run['run_id']} "
                   f"({metric}={best_run['metrics'][metric]:.4f})")
        
        return best_run
    
    def compare_runs(self, metric: str = "test_accuracy") -> List[Dict[str, Any]]:
        """
        Comparer toutes les exécutions selon une métrique.
        
        Args:
            metric: Nom de la métrique à utiliser
            
        Returns:
            Liste des exécutions triées par métrique (meilleure en premier)
        """
        runs = self.get_all_runs()
        
        # Filtrer les runs qui ont la métrique
        valid_runs = [
            run for run in runs 
            if metric in run['metrics']
        ]
        
        # Trier par métrique (décroissant)
        sorted_runs = sorted(
            valid_runs, 
            key=lambda x: x['metrics'][metric], 
            reverse=True
        )
        
        return sorted_runs
    
    def load_best_model(self, metric: str = "test_accuracy", 
                       higher_is_better: bool = True):
        """
        Charger le meilleur modèle selon une métrique.
        
        Args:
            metric: Nom de la métrique à utiliser
            higher_is_better: True si une valeur plus élevée est meilleure
            
        Returns:
            Modèle chargé ou None
        """
        best_run = self.get_best_run(metric, higher_is_better)
        
        if not best_run:
            return None
        
        return self.load_model_from_run(best_run['run_id'])
    
    def load_model_from_run(self, run_id: str):
        """
        Charger un modèle depuis une exécution spécifique.
        
        Args:
            run_id: ID de l'exécution
            
        Returns:
            Modèle chargé ou None
        """
        try:
            # Utiliser MLflow pour charger le modèle
            model_uri = f"runs:/{run_id}/models"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Modèle chargé depuis run_id: {run_id}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {run_id}: {e}")
            return None
    
    def load_model_from_registry(self, model_name: str = "valve_condition_model", 
                                 version: Optional[int] = None, 
                                 stage: Optional[str] = None):
        """
        Charger un modèle depuis le registre de modèles MLflow.
        
        Args:
            model_name: Nom du modèle enregistré
            version: Version spécifique (None pour la dernière)
            stage: Stage spécifique (None, "Staging", "Production", etc.)
            
        Returns:
            Modèle chargé ou None
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Charger la dernière version
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=None)
                if not latest_version:
                    logger.warning(f"Aucune version trouvée pour {model_name}")
                    return None
                model_uri = f"models:/{model_name}/{latest_version[0].version}"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Modèle chargé depuis le registre: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle depuis le registre: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtenir un résumé de tous les modèles disponibles.
        
        Returns:
            Dictionnaire avec le résumé des modèles
        """
        runs = self.get_all_runs()
        
        summary = {
            'total_runs': len(runs),
            'runs_with_models': sum(1 for run in runs if run['has_model']),
            'available_metrics': set(),
            'runs': []
        }
        
        for run in runs:
            if run['has_model']:
                summary['available_metrics'].update(run['metrics'].keys())
                summary['runs'].append({
                    'run_id': run['run_id'],
                    'metrics': run['metrics'],
                    'timestamp': run['tags'].get('timestamp', 'N/A')
                })
        
        summary['available_metrics'] = sorted(list(summary['available_metrics']))
        
        return summary


def select_best_model(metric: str = "test_accuracy", 
                     artifacts_dir: Path = None) -> Optional[Any]:
    """
    Fonction utilitaire pour sélectionner et charger le meilleur modèle.
    
    Args:
        metric: Métrique à utiliser pour la sélection
        artifacts_dir: Répertoire des artefacts (None pour utiliser le défaut)
        
    Returns:
        Meilleur modèle chargé ou None
    """
    selector = ModelSelector(artifacts_dir)
    return selector.load_best_model(metric)


def compare_all_models(metric: str = "test_accuracy", 
                      artifacts_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Fonction utilitaire pour comparer tous les modèles.
    
    Args:
        metric: Métrique à utiliser pour la comparaison
        artifacts_dir: Répertoire des artefacts (None pour utiliser le défaut)
        
    Returns:
        Liste des exécutions triées par métrique
    """
    selector = ModelSelector(artifacts_dir)
    return selector.compare_runs(metric)
