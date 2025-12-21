"""
Page MLflow pour l'application Streamlit
Affiche le tracking des modèles, les exécutions et les comparaisons
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.training.mlflow_manager import MLflowManager
import mlflow
from src.config.directories import directories as dirs

def display_mlflow_page():
    """Afficher la page MLflow."""
    
    st.markdown("## 🧪 MLflow - Tracking et Gestion des Modèles")
    st.markdown("Visualisez et comparez les expériences d'entraînement des modèles.")
    
    # Initialiser le gestionnaire MLflow
    mlflow_manager = MLflowManager(
        experiment_name="Valve Condition Predictor",
        tracking_uri=f"file:{dirs.raw_store_dir}"
    )
    
    # Créer des onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exécutions",
        "🏆 Meilleure Exécution",
        "📈 Comparaison",
        "⚙️ Détails"
    ])
    
    # Onglet 1: Exécutions
    with tab1:
        st.markdown("### Toutes les Exécutions")
        
        try:
            runs = mlflow_manager.get_experiment_runs()
            
            if runs.empty:
                st.info("Aucune exécution trouvée. Entraînez d'abord un modèle avec MLflow.")
            else:
                # Afficher le nombre d'exécutions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre d'Exécutions", len(runs))
                with col2:
                    st.metric("Expérience", "Valve Condition Predictor")
                with col3:
                    st.metric("Modèles Enregistrés", len(runs[runs['status'] == 'FINISHED']))
                
                # Afficher le tableau des exécutions
                st.markdown("#### Résumé des Exécutions")
                
                # Préparer les données pour l'affichage
                display_runs = []
                for idx, run in runs.iterrows():
                    display_runs.append({
                        'ID Exécution': run['run_id'][:8] + '...',
                        'Nom': run.get('tags.mlflow.runName', 'N/A'),
                        'Status': run['status'],
                        'Accuracy': f"{run.get('metrics.test_accuracy', 0):.4f}" if 'metrics.test_accuracy' in run else 'N/A',
                        'Precision': f"{run.get('metrics.test_precision', 0):.4f}" if 'metrics.test_precision' in run else 'N/A',
                        'Recall': f"{run.get('metrics.test_recall', 0):.4f}" if 'metrics.test_recall' in run else 'N/A',
                        'F1-Score': f"{run.get('metrics.test_f1', 0):.4f}" if 'metrics.test_f1' in run else 'N/A'
                    })
                
                display_df = pd.DataFrame(display_runs)
                st.dataframe(display_df, use_container_width=True)
                
                # Graphique des métriques
                st.markdown("#### Évolution des Métriques")
                
                metrics_data = []
                for idx, run in runs.iterrows():
                    if 'metrics.test_accuracy' in run and pd.notna(run['metrics.test_accuracy']):
                        metrics_data.append({
                            'Exécution': run.get('tags.mlflow.runName', run['run_id'][:8]),
                            'Accuracy': run['metrics.test_accuracy'],
                            'Precision': run.get('metrics.test_precision', 0),
                            'Recall': run.get('metrics.test_recall', 0),
                            'F1-Score': run.get('metrics.test_f1', 0)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    fig = go.Figure()
                    
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                        fig.add_trace(go.Scatter(
                            x=metrics_df['Exécution'],
                            y=metrics_df[metric],
                            mode='lines+markers',
                            name=metric,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Évolution des Métriques",
                        xaxis_title="Exécution",
                        yaxis_title="Score",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur lors du chargement des exécutions: {str(e)}")
    
    # Onglet 2: Meilleure Exécution
    with tab2:
        st.markdown("### Meilleure Exécution")
        
        try:
            best_run = mlflow_manager.get_best_run(metric="test_accuracy")
            
            if best_run:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ID Exécution", best_run['run_id'][:12] + "...")
                with col2:
                    accuracy = best_run['metrics'].get('metrics.test_accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col3:
                    st.metric("Status", "✅ Complétée")
                
                # Afficher les métriques
                st.markdown("#### Métriques")
                
                metrics_cols = st.columns(4)
                metrics_list = [
                    ('Accuracy', 'metrics.test_accuracy'),
                    ('Precision', 'metrics.test_precision'),
                    ('Recall', 'metrics.test_recall'),
                    ('F1-Score', 'metrics.test_f1')
                ]
                
                for col, (label, key) in zip(metrics_cols, metrics_list):
                    value = best_run['metrics'].get(key, 0)
                    with col:
                        st.metric(label, f"{value:.4f}")
                
                # Afficher les paramètres
                st.markdown("#### Paramètres du Modèle")
                
                params_df = pd.DataFrame([
                    {'Paramètre': k.replace('params.', ''), 'Valeur': v}
                    for k, v in best_run['params'].items()
                    if k.startswith('params.')
                ])
                
                if not params_df.empty:
                    st.dataframe(params_df, use_container_width=True)
                else:
                    st.info("Aucun paramètre enregistré")
            
            else:
                st.info("Aucune exécution trouvée")
        
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    # Onglet 3: Comparaison
    with tab3:
        st.markdown("### Comparaison des Exécutions")
        
        try:
            runs = mlflow_manager.get_experiment_runs()
            
            if len(runs) < 2:
                st.info("Au moins 2 exécutions sont nécessaires pour la comparaison")
            else:
                # Sélectionner les exécutions à comparer
                run_names = [run.get('tags.mlflow.runName', run['run_id'][:8]) for _, run in runs.iterrows()]
                selected_runs = st.multiselect(
                    "Sélectionnez les exécutions à comparer",
                    run_names,
                    default=run_names[:min(2, len(run_names))]
                )
                
                if selected_runs:
                    # Filtrer les exécutions sélectionnées
                    selected_run_ids = []
                    for _, run in runs.iterrows():
                        run_name = run.get('tags.mlflow.runName', run['run_id'][:8])
                        if run_name in selected_runs:
                            selected_run_ids.append(run['run_id'])
                    
                    comparison = mlflow_manager.compare_runs(selected_run_ids)
                    
                    # Afficher la comparaison
                    st.markdown("#### Tableau de Comparaison")
                    
                    comparison_display = []
                    for _, run in comparison.iterrows():
                        comparison_display.append({
                            'Exécution': run.get('tags.mlflow.runName', run['run_id'][:8]),
                            'Accuracy': f"{run.get('metrics.test_accuracy', 0):.4f}",
                            'Precision': f"{run.get('metrics.test_precision', 0):.4f}",
                            'Recall': f"{run.get('metrics.test_recall', 0):.4f}",
                            'F1-Score': f"{run.get('metrics.test_f1', 0):.4f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_display)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Graphique de comparaison
                    st.markdown("#### Graphique de Comparaison")
                    
                    metrics_comparison = []
                    for _, run in comparison.iterrows():
                        metrics_comparison.append({
                            'Exécution': run.get('tags.mlflow.runName', run['run_id'][:8]),
                            'Accuracy': run.get('metrics.test_accuracy', 0),
                            'Precision': run.get('metrics.test_precision', 0),
                            'Recall': run.get('metrics.test_recall', 0),
                            'F1-Score': run.get('metrics.test_f1', 0)
                        })
                    
                    metrics_comp_df = pd.DataFrame(metrics_comparison)
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Accuracy', x=metrics_comp_df['Exécution'], y=metrics_comp_df['Accuracy']),
                        go.Bar(name='Precision', x=metrics_comp_df['Exécution'], y=metrics_comp_df['Precision']),
                        go.Bar(name='Recall', x=metrics_comp_df['Exécution'], y=metrics_comp_df['Recall']),
                        go.Bar(name='F1-Score', x=metrics_comp_df['Exécution'], y=metrics_comp_df['F1-Score'])
                    ])
                    
                    fig.update_layout(
                        barmode='group',
                        title="Comparaison des Métriques",
                        xaxis_title="Exécution",
                        yaxis_title="Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    # Onglet 4: Détails
    with tab4:
        st.markdown("### Détails des Exécutions")
        
        try:
            runs = mlflow_manager.get_experiment_runs()
            
            if runs.empty:
                st.info("Aucune exécution trouvée")
            else:
                # Sélectionner une exécution
                run_names = [run.get('tags.mlflow.runName', run['run_id'][:8]) for _, run in runs.iterrows()]
                selected_run_name = st.selectbox("Sélectionnez une exécution", run_names)
                
                # Trouver l'exécution sélectionnée
                selected_run = None
                for _, run in runs.iterrows():
                    if run.get('tags.mlflow.runName', run['run_id'][:8]) == selected_run_name:
                        selected_run = run
                        break
                
                if selected_run is not None:
                    # Afficher les détails
                    st.markdown("#### Informations Générales")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ID Exécution", selected_run['run_id'][:12] + "...")
                    with col2:
                        st.metric("Status", selected_run['status'])
                    with col3:
                        st.metric("Durée", f"{(selected_run.get('end_time', selected_run['start_time']) - selected_run['start_time']) / 1000:.2f}s")
                    
                    # Afficher les métriques
                    st.markdown("#### Métriques")
                    
                    metrics_list = []
                    for col in selected_run.index:
                        if col.startswith('metrics.'):
                            metric_name = col.replace('metrics.', '')
                            metric_value = selected_run[col]
                            if pd.notna(metric_value):
                                metrics_list.append({
                                    'Métrique': metric_name,
                                    'Valeur': f"{metric_value:.4f}"
                                })
                    
                    if metrics_list:
                        metrics_df = pd.DataFrame(metrics_list)
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    # Afficher les paramètres
                    st.markdown("#### Paramètres")
                    
                    params_list = []
                    for col in selected_run.index:
                        if col.startswith('params.'):
                            param_name = col.replace('params.', '')
                            param_value = selected_run[col]
                            if pd.notna(param_value):
                                params_list.append({
                                    'Paramètre': param_name,
                                    'Valeur': param_value
                                })
                    
                    if params_list:
                        params_df = pd.DataFrame(params_list)
                        st.dataframe(params_df, use_container_width=True)
                    
                    # Afficher les tags
                    st.markdown("#### Tags")
                    
                    tags_list = []
                    for col in selected_run.index:
                        if col.startswith('tags.'):
                            tag_name = col.replace('tags.', '')
                            tag_value = selected_run[col]
                            if pd.notna(tag_value):
                                tags_list.append({
                                    'Tag': tag_name,
                                    'Valeur': tag_value
                                })
                    
                    if tags_list:
                        tags_df = pd.DataFrame(tags_list)
                        st.dataframe(tags_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    # Pied de page
    st.markdown("---")
    st.markdown("""
    ### 📚 À Propos de MLflow
    
    MLflow est une plateforme open-source pour gérer le cycle de vie du Machine Learning:
    - **Tracking**: Enregistrez les paramètres, métriques et artefacts
    - **Projects**: Empaquetez le code ML pour la réutilisation
    - **Models**: Gérez et déployez les modèles
    - **Registry**: Registre centralisé des modèles
    
    Pour visualiser les détails complets, lancez:
    ```bash
    mlflow ui --backend-store-uri file:{dirs.raw_store_dir}
    ```
    """)


if __name__ == "__main__":
    display_mlflow_page()
