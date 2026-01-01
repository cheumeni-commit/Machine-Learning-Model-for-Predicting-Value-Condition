"""
Application Streamlit Avancée pour la Prédiction de Condition de Valve
Avec Explicabilité SHAP et Monitoring du Drift avec Evidently
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Importer les modules personnalisés
from shap_explainer import load_shap_explainer
from drift_monitor import load_drift_monitor
from streamlit_mlflow_page import display_mlflow_page
from src.config.directories import directories as dirs
from src.training.model_selector import ModelSelector 
from src.read_write import load_data_train_test

# Configuration de la page
st.set_page_config(
    page_title="Valve Condition Predictor - Advanced",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger les ressources
@st.cache_resource
def load_model():
    """Charger le modèle ML entraîné."""
       # Créer une instance du sélecteur
    selector = ModelSelector()
    best_model = selector.load_best_model(metric="test_accuracy")
    return best_model

@st.cache_resource
def load_feature_importances():
    """Charger les importances des caractéristiques."""
    try:
        selector = ModelSelector()
        best_run_f1 = selector.get_best_run(metric="test_f1")
        
        if not best_run_f1 or 'run_dir' not in best_run_f1:
            return None
        
        # Extraire l'ID de l'expérience depuis le chemin du run
        experiment_id = best_run_f1['run_dir'].parent.name
        run_id = best_run_f1['run_id']
        
        # Construire le chemin vers le fichier d'importance des features
        feature_importance_path = dirs.raw_store_dir / experiment_id / run_id / "artifacts" / "feature_importance.csv"
        
        if not feature_importance_path.exists():
            return None
        
        df = pd.read_csv(feature_importance_path, names=['Feature', 'Importance'], skiprows=1, encoding='utf-8', encoding_errors='replace')

        return df.sort_values(by='Importance', ascending=False)
    except Exception as e:
        st.error(f"Erreur lors du chargement des importances des caractéristiques: {e}")
        return None

@st.cache_resource
def load_test_data():
    """Charger les données de test."""
    try:
        _, X_test, _, y_test = load_data_train_test()
        return X_test, y_test
    except:
        return None, None

@st.cache_resource
def load_train_data():
    """Charger les données d'entraînement."""
    try:
        X_train, _, y_train, __annotations__ = load_data_train_test()
        return X_train, y_train
    except:
        return None, None

@st.cache_resource
def initialize_shap_explainer(_model, _X_train, _X_test):
    """Initialiser l'explainer SHAP."""
    if _model is not None and _X_train is not None and _X_test is not None:
        return load_shap_explainer(_model, _X_train, _X_test)
    return None

@st.cache_resource
def initialize_drift_monitor(_X_train, _y_train, _X_test, _y_test):
    """Initialiser le moniteur de drift."""
    if _X_train is not None and _X_test is not None:
        return load_drift_monitor(_X_train, _y_train, _X_test, _y_test)
    return None

# Charger les ressources
model = load_model()
feature_importances = load_feature_importances()
X_test, y_test = load_test_data()
X_train, y_train = load_train_data()
shap_explainer = initialize_shap_explainer(model, X_train, X_test)
drift_monitor = initialize_drift_monitor(X_train, y_train, X_test, y_test)

# Titre et description
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>⚡ Valve Condition Predictor - Advanced</h1>
        <p style='font-size: 18px; color: #666;'>Système de Maintenance Prédictive avec Explicabilité SHAP et Monitoring du Drift</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Barre latérale
with st.sidebar:
    st.markdown("## 📋 Navigation")
    page = st.radio("Sélectionnez une page:", 
                    ["🔮 Prédiction", "📊 Analyse", "🔍 Explicabilité SHAP", "📈 Monitoring du Drift", "🧪 MLflow", "ℹ️ À Propos"])

# ============================================================================
# PAGE 1: PRÉDICTION
# ============================================================================
if page == "🔮 Prédiction":
    st.markdown("## Prédiction de Condition de Valve")
    st.markdown("Entrez les caractéristiques d'un cycle pour prédire si la valve est optimale.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Entrée des Données")
        
        # Option 1: Entrer un numéro de cycle
        st.markdown("#### Option 1: Utiliser un cycle existant")
        cycle_number = st.number_input(
            "Numéro de cycle (1-205):",
            min_value=1,
            max_value=205,
            value=100,
            step=1
        )
        
        if st.button("📈 Prédire (Cycle Existant)", key="predict_existing"):
            if X_test is not None and model is not None:
                cycle_idx = cycle_number - 1
                if 0 <= cycle_idx < len(X_test):
                    features = X_test.iloc[cycle_idx].values.reshape(1, -1)
                    actual_label = y_test.iloc[cycle_idx]
                    
                    # Prédiction
                    prediction = model.predict(features)[0]
                    prediction_proba = model.predict_proba(features)[0]
                    
                    # Afficher les résultats
                    st.markdown("### 📊 Résultats de la Prédiction")
                    
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    with col_result1:
                        if prediction == 1:
                            st.success("✅ OPTIMAL (100%)")
                        else:
                            st.warning("⚠️ NON-OPTIMAL")
                    
                    with col_result2:
                        confidence = prediction_proba[prediction]
                        st.metric("Confiance", f"{confidence*100:.1f}%")
                    
                    with col_result3:
                        if actual_label == prediction:
                            st.success("✓ Prédiction Correcte")
                        else:
                            st.error("✗ Prédiction Incorrecte")
                    
                    # Probabilités
                    st.markdown("#### Probabilités")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric("P(Non-Optimal)", f"{prediction_proba[0]*100:.2f}%")
                    
                    with prob_col2:
                        st.metric("P(Optimal)", f"{prediction_proba[1]*100:.2f}%")
                    
                    # Graphique de probabilité
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Non-Optimal', 'Optimal'],
                            y=[prediction_proba[0], prediction_proba[1]],
                            marker=dict(color=['#ff6b6b', '#51cf66']),
                            text=[f'{prediction_proba[0]*100:.1f}%', f'{prediction_proba[1]*100:.1f}%'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Distribution des Probabilités",
                        xaxis_title="Classe",
                        yaxis_title="Probabilité",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stocker le cycle sélectionné dans la session
                    st.session_state.selected_cycle = cycle_idx
                    st.session_state.selected_features = X_test.iloc[cycle_idx]
                    st.success("✓ Cycle sélectionné pour l'explicabilité SHAP")
                else:
                    st.error(f"Cycle {cycle_number} non trouvé dans les données de test.")
            else:
                st.error("Modèle ou données non chargés.")
        
        st.markdown("---")
        
        # Option 2: Entrer des caractéristiques manuellement
        st.markdown("#### Option 2: Entrer les Caractéristiques Manuellement")
        
        if model is not None:
            feature_names = X_test.columns.tolist() if X_test is not None else []
            
            if feature_names:
                # Créer des sliders pour chaque caractéristique
                features_dict = {}
                
                # Organiser les sliders en colonnes
                cols = st.columns(3)
                for idx, feature in enumerate(feature_names):
                    with cols[idx % 3]:
                        if X_test is not None:
                            min_val = X_test[feature].min()
                            max_val = X_test[feature].max() + 1
                            mean_val = X_test[feature].mean()
                        else:
                            min_val, max_val, mean_val = 0, 1, 0.50
                        
                        features_dict[feature] = st.slider(
                            feature,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(mean_val),
                            step=0.01
                        )
                
                if st.button("🔮 Prédire (Caractéristiques Manuelles)", key="predict_manual"):
                    # Créer un DataFrame avec les caractéristiques
                    features_df = pd.DataFrame([features_dict])
                    
                    # Prédiction
                    prediction = model.predict(features_df)[0]
                    prediction_proba = model.predict_proba(features_df)[0]
                    
                    # Afficher les résultats
                    st.markdown("### 📊 Résultats de la Prédiction")
                    
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        if prediction == 1:
                            st.success("✅ OPTIMAL (100%)")
                        else:
                            st.warning("⚠️ NON-OPTIMAL")
                    
                    with col_result2:
                        confidence = prediction_proba[prediction]
                        st.metric("Confiance", f"{confidence*100:.1f}%")
                    
                    # Probabilités
                    st.markdown("#### Probabilités")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric("P(Non-Optimal)", f"{prediction_proba[0]*100:.2f}%")
                    
                    with prob_col2:
                        st.metric("P(Optimal)", f"{prediction_proba[1]*100:.2f}%")

                    
                    # Graphique de probabilité
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Non-Optimal', 'Optimal'],
                            y=[prediction_proba[0], prediction_proba[1]],
                            marker=dict(color=['#ff6b6b', '#51cf66']),
                            text=[f'{prediction_proba[0]*100:.1f}%', f'{prediction_proba[1]*100:.1f}%'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Distribution des Probabilités",
                        xaxis_title="Classe",
                        yaxis_title="Probabilité",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stocker les features manuelles
                    st.session_state.manual_features = features_dict
                    st.success("✓ Caractéristiques sélectionnées pour l'explicabilité SHAP")
    
    with col2:
        st.markdown("### 📌 Informations du Modèle")
        
        st.info("""
        **Type de Modèle**
        Random Forest Classifier
        
        **Accuracy**
        
        
        **Cycles d'Entraînement**
        2000
        
        **Cycles de Test**
        205
        
        **Nombre de Caractéristiques**
        16
        """)
        
        st.markdown("### 🎯 Caractéristiques Clés")
        
        if feature_importances is not None:
            top_features = feature_importances.head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=top_features['Feature'],
                    x=top_features['Importance'],
                    orientation='h',
                    marker=dict(color='#3b82f6'),
                    text=[f'{imp*100:.1f}%' for imp in top_features['Importance']],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Top 10 Caractéristiques",
                xaxis_title="Importance",
                yaxis_title="Caractéristique",
                height=400,
                showlegend=False,
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: ANALYSE
# ============================================================================
elif page == "📊 Analyse":
    st.markdown("## Analyse du Modèle et des Données")
    
    tab1, tab2, tab3 = st.tabs(["📈 Importance des Caractéristiques", "🎯 Matrice de Confusion", "📊 Distribution des Données"])
    
    with tab1:
        st.markdown("### Importance des Caractéristiques")
        st.markdown("Les caractéristiques les plus importantes pour prédire la condition de la valve.")
        
        if feature_importances is not None:
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_importances['Importance'],
                    y=feature_importances['Feature'],
                    orientation='h',
                    marker=dict(color=feature_importances['Importance'], colorscale='Viridis'),
                    text=[f'{imp*100:.2f}%' for imp in feature_importances['Importance']],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Importance de Toutes les Caractéristiques",
                xaxis_title="Importance",
                yaxis_title="Caractéristique",
                height=600,
                showlegend=False,
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Matrice de Confusion")
        st.markdown("Performance du modèle sur l'ensemble de test.")
        
        if X_test is not None and y_test is not None and model is not None:
            from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
            
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Afficher la matrice de confusion
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Non-Optimal (Prédit)', 'Optimal (Prédit)'],
                y=['Non-Optimal (Réel)', 'Optimal (Réel)'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=True
            ))
            fig.update_layout(
                title="Matrice de Confusion",
                xaxis_title="Prédiction",
                yaxis_title="Réalité",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            
            with col2:
                precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
                st.metric("Precision (Non-Opt)", f"{precision*100:.2f}%")
            
            with col3:
                recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                st.metric("Recall (Non-Opt)", f"{recall*100:.2f}%")
            
            with col4:
                precision_opt = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                st.metric("Precision (Opt)", f"{precision_opt*100:.2f}%")
    
    with tab3:
        st.markdown("### Distribution des Données de Test")
        
        if X_test is not None:
            # Sélectionner une caractéristique à visualiser
            feature_to_plot = st.selectbox(
                "Sélectionnez une caractéristique:",
                X_test.columns.tolist()
            )
            
            fig = go.Figure()
            
            # Ajouter l'histogramme
            fig.add_trace(go.Histogram(
                x=X_test[feature_to_plot],
                nbinsx=30,
                name='Distribution',
                marker=dict(color='#3b82f6', opacity=0.7)
            ))
            
            fig.update_layout(
                title=f"Distribution de {feature_to_plot}",
                xaxis_title=feature_to_plot,
                yaxis_title="Fréquence",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: EXPLICABILITÉ SHAP
# ============================================================================
elif page == "🔍 Explicabilité SHAP":
    st.markdown("## Explicabilité SHAP - Comprendre les Prédictions")
    st.markdown("Utilisez SHAP pour comprendre pourquoi le modèle fait une prédiction spécifique.")
    
    if shap_explainer is None:
        st.error("❌ SHAP Explainer non disponible. Assurez-vous que les données sont chargées.")
    else:
        tab1, tab2, tab3 = st.tabs(["🔍 Explicabilité Locale", "🌍 Explicabilité Globale", "📊 Analyse de Dépendance"])
        
        with tab1:
            st.markdown("### Explicabilité Locale (SHAP Force Plot)")
            st.markdown("Explique pourquoi le modèle a fait une prédiction pour une instance spécifique.")
            
            # Sélectionner une instance
            instance_idx = st.slider(
                "Sélectionnez un cycle à expliquer:",
                0,
                len(X_test) - 1,
                0
            )
            
            if st.button("🔍 Générer l'explication SHAP", key="generate_shap"):
                st.markdown(f"### Explication pour le Cycle #{instance_idx + 1}")
                
                # Afficher le waterfall plot
                try:
                    fig_waterfall = shap_explainer.plot_waterfall(instance_idx)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération du waterfall plot: {e}")
                
                # Afficher le force plot
                try:
                    fig_force = shap_explainer.plot_force_plot(instance_idx)
                    st.plotly_chart(fig_force, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération du force plot: {e}")
                
                # Afficher les valeurs des features
                st.markdown("### Valeurs des Caractéristiques")
                feature_values = X_test.iloc[instance_idx]
                feature_df = pd.DataFrame({
                    'Feature': feature_values.index,
                    'Value': feature_values.values
                })
                st.dataframe(feature_df, use_container_width=True)
        
        with tab2:
            st.markdown("### Explicabilité Globale (SHAP Summary Plot)")
            st.markdown("Montre l'importance globale de chaque caractéristique sur l'ensemble des prédictions.")
            
            if st.button("🌍 Générer l'importance globale SHAP", key="generate_global_shap"):
                try:
                    fig_importance = shap_explainer.plot_global_importance()
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Afficher le tableau d'importance
                    st.markdown("### Tableau d'Importance SHAP")
                    importance_df = shap_explainer.get_global_importance()
                    st.dataframe(importance_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération de l'importance globale: {e}")
        
        with tab3:
            st.markdown("### Analyse de Dépendance")
            st.markdown("Montre la relation entre une caractéristique et sa valeur SHAP.")
            
            feature_for_dependence = st.selectbox(
                "Sélectionnez une caractéristique:",
                X_test.columns.tolist(),
                key="dependence_feature"
            )
            
            if st.button("📊 Générer le graphique de dépendance", key="generate_dependence"):
                try:
                    # Créer un graphique de dépendance simple
                    feature_idx = list(X_test.columns).index(feature_for_dependence)
                    
                    if isinstance(shap_explainer.shap_values, list):
                        shap_vals = shap_explainer.shap_values[1]
                    else:
                        shap_vals = shap_explainer.shap_values
                    
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=X_test[feature_for_dependence],
                            y=shap_vals[:, feature_idx],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=X_test[feature_for_dependence],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title=feature_for_dependence)
                            ),
                            text=X_test.index,
                            hovertemplate='<b>%{text}</b><br>' + feature_for_dependence + ': %{x:.4f}<br>SHAP Value: %{y:.4f}<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"Dependence Plot: {feature_for_dependence}",
                        xaxis_title=feature_for_dependence,
                        yaxis_title="SHAP Value",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération du graphique de dépendance: {e}")

# ============================================================================
# PAGE 4: MONITORING DU DRIFT
# ============================================================================
elif page == "📈 Monitoring du Drift":
    st.markdown("## Monitoring du Drift avec Evidently")
    st.markdown("Détectez les changements de distribution des données en production.")
    
    if drift_monitor is None:
        st.error("❌ Drift Monitor non disponible. Assurez-vous que les données sont chargées.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Résumé du Drift", "📈 Comparaison des Distributions", "🎯 Détection des Outliers", "📋 Statistiques"])
        
        with tab1:
            st.markdown("### Résumé du Drift des Données")
            st.markdown("Détecte les changements de distribution pour chaque caractéristique.")
            
            if st.button("🔍 Analyser le drift", key="analyze_drift"):
                try:
                    # Utiliser les données de test comme données actuelles
                    fig_drift = drift_monitor.plot_drift_summary(X_test)
                    st.plotly_chart(fig_drift, use_container_width=True)
                    
                    # Résumé
                    summary = drift_monitor.get_drift_summary(X_test)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total de Caractéristiques", summary['total_columns'])
                    
                    with col2:
                        st.metric("Caractéristiques avec Drift", summary['drifted_columns'])
                    
                    with col3:
                        drift_percentage = (summary['drifted_columns'] / summary['total_columns'] * 100) if summary['total_columns'] > 0 else 0
                        st.metric("Pourcentage de Drift", f"{drift_percentage:.1f}%")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse du drift: {e}")
        
        with tab2:
            st.markdown("### Comparaison des Distributions")
            st.markdown("Compare la distribution d'une caractéristique entre l'entraînement et le test.")
            
            feature_for_comparison = st.selectbox(
                "Sélectionnez une caractéristique:",
                X_test.columns.tolist(),
                key="comparison_feature"
            )
            
            if st.button("📊 Comparer les distributions", key="compare_distributions"):
                try:
                    fig_comparison = drift_monitor.plot_distribution_comparison(X_test, feature_for_comparison)
                    if fig_comparison is not None:
                        st.plotly_chart(fig_comparison, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la comparaison: {e}")
        
        with tab3:
            st.markdown("### Détection des Outliers")
            st.markdown("Détecte les outliers dans les données actuelles en utilisant la méthode IQR.")
            
            if st.button("🎯 Détecter les outliers", key="detect_outliers"):
                try:
                    fig_outliers = drift_monitor.plot_outliers_summary(X_test)
                    st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    # Afficher les détails des outliers
                    outliers = drift_monitor.detect_outliers(X_test)
                    
                    if outliers:
                        st.markdown("### Détails des Outliers")
                        for feature, indices in outliers.items():
                            st.write(f"**{feature}**: {len(indices)} outliers détectés aux indices {indices[:5]}{'...' if len(indices) > 5 else ''}")
                    else:
                        st.success("✓ Aucun outlier détecté")
                except Exception as e:
                    st.error(f"Erreur lors de la détection des outliers: {e}")
        
        with tab4:
            st.markdown("### Statistiques Comparatives")
            st.markdown("Compare les statistiques entre l'entraînement et le test.")
            
            if st.button("📋 Générer les statistiques", key="generate_stats"):
                try:
                    stats_df = drift_monitor.get_statistical_summary(X_test)
                    
                    # Afficher le tableau
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Afficher les features avec les plus grandes différences
                    st.markdown("### Features avec les Plus Grandes Différences")
                    top_diff = stats_df.nlargest(5, 'Mean_Diff_%')[['Feature', 'Mean_Diff_%', 'Std_Diff_%']]
                    st.dataframe(top_diff, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la génération des statistiques: {e}")

# ============================================================================
# PAGE 5: MLFLOW
# ============================================================================
elif page == "🧪 MLflow":
    display_mlflow_page()

# ============================================================================
# PAGE 6: À PROPOS
# ============================================================================
elif page == "ℹ️ À Propos":
    st.markdown("## À Propos du Projet")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objectif
        
        Le **Valve Condition Predictor** est un système de maintenance prédictive pour les systèmes hydrauliques industriels. 
        Il utilise le Machine Learning pour prédire si la condition d'une valve est optimale (100%) ou non, basé sur les données 
        de capteurs collectées lors de chaque cycle de production.
        
        ### 🔍 Nouvelles Fonctionnalités
        
        Cette version avancée inclut :
        
        1. **Explicabilité SHAP** - Comprendre les prédictions du modèle
           - Explicabilité locale (Force Plot, Waterfall Plot)
           - Explicabilité globale (Feature Importance)
           - Analyse de dépendance
        
        2. **Monitoring du Drift** - Détecter les changements de distribution
           - Détection du drift par feature
           - Comparaison des distributions
           - Détection des outliers
           - Statistiques comparatives
        
        ### 📊 Données
        
        Les données proviennent du jeu de données **"Condition Monitoring of Hydraulic Systems"** de l'UCI Machine Learning Repository.
        
        - **Total de cycles** : 2205
        - **Cycles d'entraînement** : 2000
        - **Cycles de test** : 205
        - **Capteurs** :
          - PS2 (Pression) - 100 Hz
          - FS1 (Débit volumique) - 10 Hz
        
        ### 🤖 Modèle
        
        **Type** : Random Forest Classifier
        - **Nombre d'arbres** : 100
        - **Accuracy** : 
        - **Caractéristiques** : 16 (statistiques extraites des capteurs)
        
        ### 🚀 Utilisation
        
        1. Allez à l'onglet **"Prédiction"**
        2. Entrez un numéro de cycle (1-205) ou les caractéristiques manuellement
        3. Cliquez sur **"Prédire"**
        4. Consultez le résultat et la confiance du modèle
        5. Allez à **"Explicabilité SHAP"** pour comprendre la prédiction
        6. Allez à **"Monitoring du Drift"** pour surveiller la qualité des données
        """)
    
    with col2:
        st.markdown("### 📌 Informations Techniques")
        
        st.info("""
        **Stack Technologique**
        
        - **Backend** : Python, scikit-learn
        - **Frontend** : Streamlit
        - **Explicabilité** : SHAP
        - **Monitoring** : Evidently
        - **Visualisation** : Plotly
        - **Données** : pandas, numpy
        
        **Modèle ML**
        
        - Type : Random Forest
        - Accuracy : 
        - Cycles d'entraînement : 2000
        - Cycles de test : 205
        
        **Caractéristiques**
        
        - 18 caractéristiques statistiques
        - 2 capteurs (PS2, FS1)
        - Extraction automatique
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px;'>
    <p>Valve Condition Predictor - Advanced © 2025 | Avec Explicabilité SHAP, Monitoring du Drift et MLflow</p>
</div>
""", unsafe_allow_html=True)
