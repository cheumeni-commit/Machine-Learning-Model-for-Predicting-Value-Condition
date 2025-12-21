"""
Module pour l'explicabilité SHAP
Explique les prédictions du modèle avec SHAP (locale et globale)
"""

import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict, Any
import os

class SHAPExplainer:
    """Classe pour expliquer les prédictions avec SHAP."""
    
    def __init__(self, model, X_train, X_test):
        """
        Initialiser l'explainer SHAP.
        
        Args:
            model: Modèle entraîné
            X_train: Données d'entraînement
            X_test: Données de test
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self):
        """Initialiser l'explainer SHAP avec TreeExplainer."""
        if self.explainer is None:
            # Utiliser TreeExplainer pour Random Forest
            self.explainer = shap.TreeExplainer(self.model)
            # Calculer les SHAP values pour les données de test
            self.shap_values = self.explainer.shap_values(self.X_test)
            
    def get_global_importance(self) -> pd.DataFrame:
        """
        Obtenir l'importance globale des caractéristiques (SHAP).
        
        Returns:
            DataFrame avec l'importance globale
        """
        if self.shap_values is None:
            self.initialize_explainer()
        
        # Calculer la moyenne absolue des SHAP values pour chaque classe
        # Pour Random Forest avec 2 classes, shap_values est une liste
        try:
            if isinstance(self.shap_values, list):
                # Prendre les SHAP values pour la classe "Optimal" (index 1)
                shap_vals = np.array(self.shap_values[1])
                if len(shap_vals.shape) == 3:  # (n_samples, n_features, n_outputs)
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0).mean(axis=1)
                else:
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            else:
                shap_vals = np.array(self.shap_values)
                if len(shap_vals.shape) == 3:
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0).mean(axis=1)
                else:
                    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'Feature': self.X_test.columns,
                'SHAP_Importance': mean_abs_shap
            }).sort_values('SHAP_Importance', ascending=False)
            
            return importance_df
        except Exception as e:
            # Retourner une importance basée sur le modèle en cas d'erreur
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': self.X_test.columns,
                    'SHAP_Importance': self.model.feature_importances_
                }).sort_values('SHAP_Importance', ascending=False)
                return importance_df
            raise e
    
    def get_local_explanation(self, instance_idx: int) -> Dict[str, Any]:
        """
        Obtenir l'explication locale pour une instance.
        
        Args:
            instance_idx: Index de l'instance
            
        Returns:
            Dictionnaire avec les SHAP values et les valeurs des caractéristiques
        """
        if self.shap_values is None:
            self.initialize_explainer()
        
        if isinstance(self.shap_values, list):
            # Prendre les SHAP values pour la classe "Optimal" (index 1)
            shap_vals = self.shap_values[1][instance_idx]
        else:
            shap_vals = self.shap_values[instance_idx]
        
        feature_values = self.X_test.iloc[instance_idx].values
        
        return {
            'shap_values': shap_vals,
            'feature_values': feature_values,
            'features': self.X_test.columns,
            'base_value': self.explainer.expected_value if isinstance(self.explainer.expected_value, (int, float)) else self.explainer.expected_value[1]
        }
    
    def get_force_plot_data(self, instance_idx: int) -> Dict[str, Any]:
        """
        Obtenir les données pour un force plot.
        
        Args:
            instance_idx: Index de l'instance
            
        Returns:
            Dictionnaire avec les données pour le force plot
        """
        explanation = self.get_local_explanation(instance_idx)
        
        # Trier par importance (valeur absolue des SHAP values)
        importance_idx = np.argsort(np.abs(explanation['shap_values']))[::-1]
        
        top_features = []
        for idx in importance_idx[:10]:  # Top 10 features
            top_features.append({
                'feature': explanation['features'][idx],
                'value': explanation['feature_values'][idx],
                'shap': explanation['shap_values'][idx]
            })
        
        return {
            'base_value': explanation['base_value'],
            'top_features': top_features,
            'total_shap': np.sum(explanation['shap_values'])
        }
    
    def plot_waterfall(self, instance_idx: int, max_features: int = 10):
        """
        Créer un graphique waterfall pour l'explication locale.
        
        Args:
            instance_idx: Index de l'instance
            max_features: Nombre maximum de features à afficher
        """
        force_data = self.get_force_plot_data(instance_idx)
        
        # Préparer les données pour le graphique
        features = ['Base Value'] + [f['feature'] for f in force_data['top_features']]
        values = [force_data['base_value']] + [f['shap'] for f in force_data['top_features']]
        
        # Créer le graphique waterfall
        fig = go.Figure(go.Waterfall(
            x=features,
            y=values,
            connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            increasing={"marker": {"color": "#51cf66"}},
            decreasing={"marker": {"color": "#ff6b6b"}},
            totals={"marker": {"color": "#3b82f6"}}
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot - Cycle #{instance_idx + 1}",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_force_plot(self, instance_idx: int):
        """
        Créer un force plot pour l'explication locale.
        
        Args:
            instance_idx: Index de l'instance
        """
        force_data = self.get_force_plot_data(instance_idx)
        
        # Séparer les features positives et négatives
        positive_features = [f for f in force_data['top_features'] if f['shap'] > 0]
        negative_features = [f for f in force_data['top_features'] if f['shap'] < 0]
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les features positives
        if positive_features:
            fig.add_trace(go.Bar(
                y=[f['feature'] for f in positive_features],
                x=[f['shap'] for f in positive_features],
                orientation='h',
                name='Augmente la prédiction',
                marker=dict(color='#51cf66'),
                text=[f"{f['shap']:.4f}" for f in positive_features],
                textposition='auto'
            ))
        
        # Ajouter les features négatives
        if negative_features:
            fig.add_trace(go.Bar(
                y=[f['feature'] for f in negative_features],
                x=[f['shap'] for f in negative_features],
                orientation='h',
                name='Diminue la prédiction',
                marker=dict(color='#ff6b6b'),
                text=[f"{f['shap']:.4f}" for f in negative_features],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=f"SHAP Force Plot - Cycle #{instance_idx + 1}",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            height=400,
            barmode='relative'
        )
        
        return fig
    
    def plot_global_importance(self):
        """Créer un graphique d'importance globale SHAP."""
        try:
            importance_df = self.get_global_importance()
            
            # Normaliser les valeurs pour la colorscale
            norm_importance = (importance_df['SHAP_Importance'] - importance_df['SHAP_Importance'].min()) / (importance_df['SHAP_Importance'].max() - importance_df['SHAP_Importance'].min())
            
            fig = go.Figure(data=[
                go.Bar(
                    y=importance_df['Feature'],
                    x=importance_df['SHAP_Importance'],
                    orientation='h',
                    marker=dict(
                        color=norm_importance,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f'{val:.4f}' for val in importance_df['SHAP_Importance']],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="SHAP Global Feature Importance",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Feature",
                height=600,
                showlegend=False,
                margin=dict(l=150)
            )
            
            return fig
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique: {str(e)}")
            return None
    
    def plot_summary(self):
        """Créer un summary plot SHAP."""
        if self.shap_values is None:
            self.initialize_explainer()
        
        # Utiliser les SHAP values pour la classe "Optimal"
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        # Créer le summary plot
        fig = shap.summary_plot(
            shap_vals,
            self.X_test,
            plot_type="bar",
            show=False
        )
        
        return fig
    
    def plot_dependence(self, feature_name: str):
        """
        Créer un dependence plot pour une feature.
        
        Args:
            feature_name: Nom de la feature
        """
        if self.shap_values is None:
            self.initialize_explainer()
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        feature_idx = list(self.X_test.columns).index(feature_name)
        
        fig = shap.dependence_plot(
            feature_idx,
            shap_vals,
            self.X_test,
            show=False
        )
        
        return fig


def load_shap_explainer(model, X_train, X_test) -> SHAPExplainer:
    """Charger et initialiser l'explainer SHAP."""
    explainer = SHAPExplainer(model, X_train, X_test)
    explainer.initialize_explainer()
    return explainer
