"""
Exemple d'utilisation du sélecteur de modèle pour choisir le meilleur modèle
"""

from pathlib import Path
from src.training.model_selector import ModelSelector, select_best_model, compare_all_models

def main():
    """Exemple d'utilisation du sélecteur de modèle."""
    
    # Créer une instance du sélecteur
    selector = ModelSelector()
    
    # 1. Obtenir un résumé de tous les modèles
    print("=" * 60)
    print("RÉSUMÉ DES MODÈLES DISPONIBLES")
    print("=" * 60)
    summary = selector.get_model_summary()
    print(f"Total des exécutions: {summary['total_runs']}")
    print(f"Exécutions avec modèles: {summary['runs_with_models']}")
    print(f"Métriques disponibles: {', '.join(summary['available_metrics'])}")
    print()
    
    # 2. Comparer tous les modèles par test_accuracy
    print("=" * 60)
    print("COMPARAISON DES MODÈLES (par test_accuracy)")
    print("=" * 60)
    sorted_runs = selector.compare_runs(metric="test_accuracy")
    for i, run in enumerate(sorted_runs, 1):
        print(f"\n{i}. Run ID: {run['run_id']}")
        print(f"   Test Accuracy: {run['metrics'].get('test_accuracy', 'N/A'):.4f}")
        print(f"   Test F1: {run['metrics'].get('test_f1', 'N/A'):.4f}")
        print(f"   Test Precision: {run['metrics'].get('test_precision', 'N/A'):.4f}")
        print(f"   Test Recall: {run['metrics'].get('test_recall', 'N/A'):.4f}")
    print()
    
    # 3. Obtenir le meilleur modèle selon test_accuracy
    print("=" * 60)
    print("MEILLEUR MODÈLE (par test_accuracy)")
    print("=" * 60)
    best_run = selector.get_best_run(metric="test_accuracy")
    if best_run:
        print(f"Run ID: {best_run['run_id']}")
        print(f"Métriques:")
        for metric, value in best_run['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print()
        
        # Charger le meilleur modèle
        print("Chargement du meilleur modèle...")
        model = selector.load_best_model(metric="test_accuracy")
        print(model)
        if model:
            print(f"✓ Modèle chargé avec succès!")
            print(f"  Type: {type(model).__name__}")
    print()
    
    # 4. Obtenir le meilleur modèle selon test_f1
    print("=" * 60)
    print("MEILLEUR MODÈLE (par test_f1)")
    print("=" * 60)
    best_run_f1 = selector.get_best_run(metric="test_f1")
    if best_run_f1:
        print(f"Run ID: {best_run_f1['run_id']}")
        print(f"Test F1: {best_run_f1['metrics'].get('test_f1', 'N/A'):.4f}")
        print()
    
    # 5. Utiliser la fonction utilitaire simple
    print("=" * 60)
    print("UTILISATION DE LA FONCTION UTILITAIRE")
    print("=" * 60)
    model = select_best_model(metric="test_accuracy")
    if model:
        print("✓ Modèle chargé avec la fonction utilitaire")
    
    # 6. Charger depuis le registre de modèles (si disponible)
    print("=" * 60)
    print("CHARGEMENT DEPUIS LE REGISTRE")
    print("=" * 60)
    try:
        model_from_registry = selector.load_model_from_registry(
            model_name="valve_condition_model",
            version=None  # Dernière version
        )
        if model_from_registry:
            print("✓ Modèle chargé depuis le registre")
    except Exception as e:
        print(f"⚠ Impossible de charger depuis le registre: {e}")


if __name__ == "__main__":
    main()
