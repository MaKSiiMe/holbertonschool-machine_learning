#!/usr/bin/env python3
"""
Dashboard interactif pour l'analyse des expériences CIFAR-10.
Ce script utilise Streamlit et Plotly pour créer une application web
permettant de visualiser, filtrer et comparer les résultats des
entraînements stockés dans des fichiers JSON.

Usage :
1. Assurez-vous d'avoir les bibliothèques :
   pip install streamlit pandas plotly scikit-learn
2. Lancez l'application depuis votre terminal :
   streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import glob
import json
from pathlib import Path

# --- Configuration de la Page Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard CIFAR-10")

# --- Fonctions de Traitement des Données ---

def g(d, keys, default=None):
    """Accesseur sécurisé pour les dictionnaires imbriqués."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

@st.cache_data
def load_and_process_data(files_pattern):
    """
    Charge tous les JSON, les traite et les retourne sous forme de deux
    DataFrames Pandas optimisés pour l'analyse.
    """
    files = glob.glob(files_pattern)
    if not files:
        return pd.DataFrame(), pd.DataFrame()

    all_runs_summary = []
    all_runs_curves = []

    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
                config = g(d, ["config"], {})
                run_id = Path(fp).stem

                summary = {
                    'run_id': run_id,
                    'unfreeze': g(config, ["n_unfreeze"], -1),
                    'augment': g(config, ["augment"], "none"),
                    'optimizer': g(config, ["optimizer"], "sgd"),
                    'lr_stage2': g(config, ["lr_stage2"], 0.0),
                    'val_accuracy': g(d, ["metrics", "val_end2end"]),
                    'test_accuracy': g(d, ["metrics", "test_end2end"]),
                    'time_total': g(d, ["timing_sec", "total"]),
                    'confusion_matrix': g(d, ["confusion_matrix"])
                }
                all_runs_summary.append(summary)

                for stage in [1, 2]:
                    for metric in ["accuracy", "loss"]:
                        for curve_prefix, curve_type in [("", "Training"), ("val_", "Validation")]:
                            curve_key = f"{curve_prefix}{metric}"
                            curve_values = g(d, ["curves", f"stage{stage}", curve_key], [])
                            
                            for epoch, value in enumerate(curve_values):
                                curves_data = summary.copy()
                                curves_data.update({
                                    'stage': stage,
                                    'metric': metric,
                                    'epoch': epoch + 1,
                                    'value': value,
                                    'type': curve_type
                                })
                                all_runs_curves.append(curves_data)
        except Exception as e:
            st.warning(f"Impossible de charger ou traiter {fp}: {e}")

    summary_df = pd.DataFrame(all_runs_summary).dropna(subset=['val_accuracy'])
    curves_df = pd.DataFrame(all_runs_curves)
    return summary_df, curves_df

# --- Interface du Dashboard ---

st.title("🔬 Dashboard d'Analyse pour CIFAR-10")
st.markdown("Application interactive pour explorer les résultats des expériences de transfer learning.")

# --- Panneau de Contrôle (Sidebar) ---
with st.sidebar:
    st.header("🎛️ Panneau de Contrôle")

    # --- MODIFICATION : Calcul automatique du chemin ---
    # Chemin du script actuel (dashboard_app.py)
    script_path = Path(__file__).resolve()
    # Chemin du dossier "transfer_learning" (en remontant de 2 niveaux depuis .../tools/dashboard_app.py)
    project_root = script_path.parent.parent
    # Chemin du dossier results
    results_path = project_root / "results"
    files_pattern = str(results_path / "*.json")

    st.markdown("Chemin des fichiers JSON détecté :")
    st.info(f"{files_pattern}")
    # --- FIN DE LA MODIFICATION ---
    
    summary_df, curves_df = load_and_process_data(files_pattern)

    if summary_df.empty:
        st.warning("Aucun fichier JSON trouvé dans le dossier détecté. Avez-vous lancé un entraînement ?")
        st.stop()
    
    st.subheader("Filtres des Expériences")
    unfreeze_options = sorted(summary_df['unfreeze'].unique())
    selected_unfreeze = st.multiselect("Niveaux de Fine-Tuning (unfreeze)", unfreeze_options, default=unfreeze_options)

    optimizer_options = sorted(summary_df['optimizer'].unique())
    selected_optimizers = st.multiselect("Optimiseurs", optimizer_options, default=optimizer_options)

    augment_options = sorted(summary_df['augment'].unique())
    selected_augment = st.multiselect("Data Augmentation", augment_options, default=augment_options)

    st.subheader("Options des Graphiques")
    selected_stage = st.selectbox("Étape (Stage)", [1, 2], index=1) # Default to stage 2
    selected_metric = st.selectbox("Métrique", ["accuracy", "loss"], index=0)

# Filtrage des DataFrames
filtered_summary = summary_df[
    summary_df['unfreeze'].isin(selected_unfreeze) &
    summary_df['optimizer'].isin(selected_optimizers) &
    summary_df['augment'].isin(selected_augment)
]

filtered_curves = curves_df[
    curves_df['run_id'].isin(filtered_summary['run_id'])
]

# --- Affichage des Graphiques ---

if filtered_summary.empty:
    st.warning("Aucune expérience ne correspond aux filtres sélectionnés.")
else:
    st.header("📊 Vue d'Ensemble des Expériences Filtrées")
    st.markdown(f"**{len(filtered_summary)} expériences** correspondent à votre sélection.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance vs. Coût")
        fig_scatter = px.scatter(
            filtered_summary,
            x="val_accuracy",
            y="time_total",
            color="optimizer",
            size="unfreeze",
            hover_data=['run_id', 'unfreeze', 'augment', 'lr_stage2'],
            title="Vue d'Ensemble Interactive des Expériences"
        )
        fig_scatter.update_layout(xaxis_title="Accuracy de Validation Finale", yaxis_title="Temps d'Entraînement Total (s)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Courbes d'Apprentissage")
        curves_to_plot = filtered_curves[
            (filtered_curves['stage'] == selected_stage) & 
            (filtered_curves['metric'] == selected_metric)
        ]
        
        if not curves_to_plot.empty:
            fig_curves = px.line(
                curves_to_plot,
                x="epoch",
                y="value",
                color="run_id",
                line_dash="type",
                title=f"{selected_metric.capitalize()} (Stage {selected_stage})",
                labels={'value': selected_metric.capitalize(), 'epoch': 'Epoch', 'type': 'Type'},
                category_orders={"type": ["Training", "Validation"]}
            )
            st.plotly_chart(fig_curves, use_container_width=True)
        else:
            st.info("Aucune courbe d'apprentissage disponible pour cette sélection.")

    st.header("🏆 Analyse du Meilleur Modèle de la Sélection")
    
    best_run_summary = filtered_summary.loc[filtered_summary['val_accuracy'].idxmax()]
    
    st.markdown(f"Le meilleur modèle de la sélection est **{best_run_summary['run_id']}** avec une accuracy de validation de **{best_run_summary['val_accuracy']:.4f}**.")
    st.write("Détails de la configuration :", best_run_summary.to_dict())

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Matrice de Confusion")
        cm_info = best_run_summary.get('confusion_matrix', {})
        cm_data = None
        if isinstance(cm_info, dict):
            shape = cm_info.get("shape")
            data = cm_info.get("data")
            if shape and data:
                cm_data = np.array(data).reshape(shape)
        elif isinstance(cm_info, list):
             cm_data = np.array(cm_info)

        if cm_data is not None:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            fig_cm = px.imshow(
                cm_data,
                labels=dict(x="Prédiction", y="Vraie Valeur", color="Nombre"),
                x=class_names,
                y=class_names,
                text_auto=True,
                title=f"Matrice de Confusion pour {best_run_summary['run_id']}"
            )
            fig_cm.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Matrice de confusion non disponible pour ce modèle.")
            
    with col4:
        st.subheader("Analyses Futures")
        st.info("Cet espace peut être utilisé pour des visualisations avancées comme Grad-CAM ou l'analyse des erreurs.")