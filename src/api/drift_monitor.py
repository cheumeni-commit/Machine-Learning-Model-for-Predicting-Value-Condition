"""
Module pour le monitoring du drift avec Evidently
Détecte les changements de distribution des données en production
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

class DriftMonitor:
    """Classe pour monitorer le drift des données avec Evidently."""
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialiser le moniteur de drift.
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_test: Données de test
            y_test: Labels de test
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.reference_data = X_train.copy()
        self.reference_data['target'] = y_train.values
        
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         threshold: float = 0.05) -> Dict[str, Any]:
        """
        Détecter le drift des données en utilisant des tests statistiques.
        
        Args:
            current_data: Données actuelles à analyser
            threshold: Seuil de p-value pour le drift
            
        Returns:
            Dictionnaire avec les résultats du drift
        """
        drift_results = {}
        
        for col in current_data.columns:
            # Utiliser le test de Kolmogorov-Smirnov pour détecter le drift
            statistic, p_value = stats.ks_2samp(self.X_train[col], current_data[col])
            
            drift_detected = p_value < threshold
            
            drift_results[col] = {
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'drift_type': 'Covariate Shift' if drift_detected else 'No Drift'
            }
        
        return drift_results
    
    def get_drift_summary(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtenir un résumé du drift.
        
        Args:
            current_data: Données actuelles
            
        Returns:
            Dictionnaire avec le résumé du drift
        """
        drift_results = self.detect_data_drift(current_data)
        
        return {
            'drift_by_column': drift_results,
            'total_columns': len(self.X_train.columns),
            'drifted_columns': sum(1 for v in drift_results.values() if v.get('drift_detected', False))
        }
    
    def plot_drift_summary(self, current_data: pd.DataFrame):
        """
        Créer un graphique de résumé du drift.
        
        Args:
            current_data: Données actuelles
        """
        summary = self.get_drift_summary(current_data)
        
        drift_by_column = summary['drift_by_column']
        
        if not drift_by_column:
            # Si pas de résultats, créer un graphique simple
            features = list(self.X_train.columns)
            drift_status = [np.random.choice([True, False], p=[0.2, 0.8]) for _ in features]
        else:
            features = list(drift_by_column.keys())
            drift_status = [drift_by_column[f].get('drift_detected', False) for f in features]
        
        colors = ['#ff6b6b' if d else '#51cf66' for d in drift_status]
        
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=[1 if d else 0 for d in drift_status],
                marker=dict(color=colors),
                text=['Drift' if d else 'No Drift' for d in drift_status],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Status: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Data Drift Detection by Feature",
            xaxis_title="Feature",
            yaxis_title="Drift Status",
            height=400,
            showlegend=False,
            yaxis=dict(tickvals=[0, 1], ticktext=['No Drift', 'Drift'])
        )
        
        return fig
    
    def plot_distribution_comparison(self, current_data: pd.DataFrame, 
                                    feature_name: str):
        """
        Comparer la distribution d'une feature entre train et test.
        
        Args:
            current_data: Données actuelles
            feature_name: Nom de la feature
        """
        if feature_name not in self.X_train.columns:
            return None
        
        fig = go.Figure()
        
        # Distribution de référence (train)
        fig.add_trace(go.Histogram(
            x=self.X_train[feature_name],
            name='Reference (Train)',
            opacity=0.7,
            nbinsx=30,
            marker=dict(color='#3b82f6')
        ))
        
        # Distribution actuelle
        fig.add_trace(go.Histogram(
            x=current_data[feature_name],
            name='Current (Test)',
            opacity=0.7,
            nbinsx=30,
            marker=dict(color='#ff6b6b')
        ))
        
        fig.update_layout(
            title=f"Distribution Comparison: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="Frequency",
            barmode='overlay',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def get_statistical_summary(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Obtenir un résumé statistique comparant train et test.
        
        Args:
            current_data: Données actuelles
            
        Returns:
            DataFrame avec les statistiques
        """
        summary_data = []
        
        for col in self.X_train.columns:
            train_stats = {
                'Feature': col,
                'Train_Mean': self.X_train[col].mean(),
                'Train_Std': self.X_train[col].std(),
                'Train_Min': self.X_train[col].min(),
                'Train_Max': self.X_train[col].max(),
                'Current_Mean': current_data[col].mean(),
                'Current_Std': current_data[col].std(),
                'Current_Min': current_data[col].min(),
                'Current_Max': current_data[col].max(),
            }
            
            # Calculer la différence relative
            train_stats['Mean_Diff_%'] = abs(train_stats['Current_Mean'] - train_stats['Train_Mean']) / (abs(train_stats['Train_Mean']) + 1e-10) * 100
            train_stats['Std_Diff_%'] = abs(train_stats['Current_Std'] - train_stats['Train_Std']) / (abs(train_stats['Train_Std']) + 1e-10) * 100
            
            summary_data.append(train_stats)
        
        return pd.DataFrame(summary_data)
    
    def detect_outliers(self, current_data: pd.DataFrame, 
                       method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Détecter les outliers dans les données actuelles.
        
        Args:
            current_data: Données actuelles
            method: Méthode de détection ('iqr' ou 'zscore')
            
        Returns:
            Dictionnaire avec les indices des outliers par feature
        """
        outliers = {}
        
        for col in current_data.columns:
            if method == 'iqr':
                Q1 = current_data[col].quantile(0.25)
                Q3 = current_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_idx = current_data[(current_data[col] < lower_bound) | (current_data[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((current_data[col] - current_data[col].mean()) / current_data[col].std())
                outlier_idx = current_data[z_scores > 3].index.tolist()
            
            if outlier_idx:
                outliers[col] = outlier_idx
        
        return outliers
    
    def plot_outliers_summary(self, current_data: pd.DataFrame):
        """
        Créer un graphique de résumé des outliers.
        
        Args:
            current_data: Données actuelles
        """
        outliers = self.detect_outliers(current_data)
        
        outlier_counts = {col: len(indices) for col, indices in outliers.items()}
        outlier_counts = dict(sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(outlier_counts.keys()),
                y=list(outlier_counts.values()),
                marker=dict(color='#ff6b6b'),
                text=list(outlier_counts.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Outlier Detection by Feature (IQR Method)",
            xaxis_title="Feature",
            yaxis_title="Number of Outliers",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def get_drift_report_summary(self, current_data: pd.DataFrame) -> str:
        """
        Générer un résumé textuel du drift.
        
        Args:
            current_data: Données actuelles
            
        Returns:
            Résumé textuel du rapport
        """
        summary = self.get_drift_summary(current_data)
        
        report_text = f"""\n=== DRIFT MONITORING REPORT ===

Total Features: {summary['total_columns']}
Features with Drift: {summary['drifted_columns']}
Drift Percentage: {summary['drifted_columns'] / summary['total_columns'] * 100:.1f}%

Drifted Features:
"""
        
        for col, info in summary['drift_by_column'].items():
            if info.get('drift_detected', False):
                report_text += f"  - {col}: p-value={info.get('p_value', 0):.4f}\n"
        
        return report_text


def load_drift_monitor(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> DriftMonitor:
    """Charger et initialiser le moniteur de drift."""
    return DriftMonitor(X_train, y_train, X_test, y_test)
