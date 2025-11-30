# 📈 Analyse des Marchés Financiers & Prédiction de Prix avec Python & IA

## 👨‍💻 Auteur
**Mohamed Zaari**  
CSES
## 📖 Aperçu du Projet
Une plateforme complète d'analyse financière développée en Python pour l'analyse de portefeuille, l'optimisation d'investissements et la prédiction de prix actions grâce à l'intelligence artificielle. Ce projet démontre un pipeline complet de data science, de l'acquisition de données à la modélisation prédictive et la visualisation.

**Note :** Ce repository est conçu à des **fins éducatives** pour présenter des compétences en Data Science, IA, Analyse Financière et programmation Python.

## ✨ Fonctionnalités Principales

### 🔍 Analyse des Données
- **Acquisition de Données** : Téléchargement de données marché en temps réel via Yahoo Finance
- **Analyse Technique** : Calcul des rendements, volatilité, matrices de corrélation
- **Métriques de Performance** : Sharpe Ratio, rendements annualisés, analyse de risque

### 🤖 Intelligence Artificielle
- **Prédiction de Prix** : Réseaux de neurones (MLPRegressor) pour la prévision à 30 jours
- **Optimisation de Portefeuille** : Maximisation du ratio de Sharpe via la théorie moderne du portefeuille
- **Modèles ML** : Standardisation des données et entraînement de modèles supervisés

### 📊 Visualisation & Reporting
- **Graphiques Interactifs** : Évolution des prix, rendements cumulés, matrices de corrélation
- **Rapports Automatisés** : Génération de PDF avec analyses détaillées
- **Dashboard Web** : Interface Streamlit pour l'analyse interactive

## 🛠️ Technologies Utilisées

### 📈 Data Science & Analyse
- **Pandas** - Manipulation et analyse de données financières
- **NumPy** - Calculs numériques et optimisations mathématiques
- **Matplotlib** - Visualisation de données et création de graphiques

### 🤖 Machine Learning & IA
- **Scikit-learn** - MLPRegressor pour les prédictions, StandardScaler pour le preprocessing
- **Théorie du Portefeuille** - Optimisation Markowitz, ratio de Sharpe

### 🌐 Data & Sécurité
- **yfinance** - Intégration avec l'API Yahoo Finance
- **Certifi** - Gestion des certificats SSL pour les requêtes sécurisées

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.7 ou supérieur
- Gestionnaire de packages pip

### Installation & Utilisation

```bash
# Cloner le repository
git clone https://github.com/Zaari-Mohamed/Projet-AI.git

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application web
streamlit run app.py