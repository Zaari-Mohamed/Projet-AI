import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Analyse Financière IA",
    page_icon="📈",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🤖 Analyse Financière avec Intelligence Artificielle</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres d'analyse")
    
    tickers_input = st.text_input("Tickers (séparés par des virgules):", "AAPL,MSFT,GOOGL")
    start_date = st.date_input("Date de début:", datetime(2023, 1, 1))
    end_date = st.date_input("Date de fin:", datetime(2024, 1, 1))
    
    # Paramètres avancés
    with st.expander("Paramètres avancés"):
        future_days = st.slider("Jours de prédiction:", 7, 90, 30)
        num_portfolios = st.slider("Simulations portfolio:", 1000, 20000, 5000)
    
    analyze_btn = st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True)

# Fonctions d'analyse
def download_yahoo_data(tickers, start, end):
    """Télécharge les données depuis Yahoo Finance - CORRIGÉE"""
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    try:
        # Télécharger toutes les données
        data = yf.download(ticker_list, start=start, end=end)
        
        # Vérifier si le téléchargement a réussi
        if data.empty:
            st.error("❌ Aucune donnée trouvée. Vérifiez les tickers et la période.")
            return None, ticker_list
        
        # CORRECTION : Gérer les différents noms de colonnes
        if isinstance(data.columns, pd.MultiIndex):
            # Format multi-index (pour plusieurs tickers)
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                price_data = data['Close']
                st.info("ℹ️ Utilisation des prix de clôture (Adj Close non disponible)")
            else:
                st.error("❌ Colonnes de prix non trouvées")
                return None, ticker_list
        else:
            # Format simple index (pour un seul ticker)
            if 'Adj Close' in data.columns:
                price_data = data['Adj Close']
            elif 'Close' in data.columns:
                price_data = data['Close']
                st.info("ℹ️ Utilisation des prix de clôture (Adj Close non disponible)")
            else:
                st.error("❌ Colonnes de prix non trouvées")
                return None, ticker_list
        
        # Vérifier que nous avons des données
        if price_data.empty:
            st.error("❌ Aucune donnée de prix trouvée")
            return None, ticker_list
            
        # Vérifier les tickers manquants
        if isinstance(price_data, pd.DataFrame):
            missing_tickers = set(ticker_list) - set(price_data.columns)
        else:
            # Cas d'un seul ticker (Series)
            missing_tickers = set(ticker_list) - set([ticker_list[0]])
            
        if missing_tickers:
            st.warning(f"⚠️ Tickers non trouvés: {missing_tickers}")
            # Garder seulement les tickers disponibles
            if isinstance(price_data, pd.DataFrame):
                available_tickers = [t for t in ticker_list if t in price_data.columns]
            else:
                available_tickers = ticker_list[:1]  # Garder le premier ticker
        else:
            available_tickers = ticker_list
            
        if len(available_tickers) == 0:
            st.error("❌ Aucun ticker valide trouvé")
            return None, ticker_list
            
        # S'assurer que nous retournons un DataFrame
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()
            price_data.columns = available_tickers
            
        return price_data, available_tickers
        
    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement: {str(e)}")
        return None, ticker_list

def calculate_statistics(data):
    """Calcule les statistiques de base"""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    volatility = returns.std()
    correlation = returns.corr()
    cumulative_returns = (1 + returns).cumprod()
    
    return returns, mean_returns, volatility, correlation, cumulative_returns

def optimize_portfolio(returns, tickers, num_portfolios=10000):
    """Optimisation du portefeuille - CORRIGÉE"""
    if len(tickers) == 0:
        return np.array([]), np.array([0, 0, 0])
    
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # CORRECTION : S'assurer que mean_returns_annual a la bonne dimension
    mean_returns_annual = returns.mean() * 252
    cov_matrix_annual = returns.cov() * 252

    for i in range(num_portfolios):
        # Générer des poids aléatoires
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)  # Normaliser à 1
        
        # CORRECTION : Vérifier les dimensions
        if len(weights) != len(mean_returns_annual):
            st.error(f"Dimension mismatch: weights {len(weights)} vs returns {len(mean_returns_annual)}")
            continue
            
        port_return = np.dot(weights, mean_returns_annual)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
        sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0

        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio
        weights_record.append(weights)

    if len(weights_record) == 0:
        return np.ones(len(tickers)) / len(tickers), np.array([0, 0, 0])
    
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    optimal_results = results[:, max_sharpe_idx]

    return optimal_weights, optimal_results

def predict_prices(data, tickers, future_days=30):
    """Prédictions avec ML"""
    predictions_dict = {}
    
    for ticker in tickers:
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[ticker].values

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,  # Réduit pour plus de rapidité
                random_state=42,
                alpha=0.001,
                learning_rate_init=0.001
            )

            model.fit(X_scaled, y_scaled)

            future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
            future_X_scaled = scaler_X.transform(future_X)
            pred_scaled = model.predict(future_X_scaled)
            predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

            predictions_dict[ticker] = predictions

        except Exception as e:
            st.warning(f"Prédiction simple pour {ticker}: {str(e)}")
            predictions_dict[ticker] = np.full(future_days, data[ticker].iloc[-1])
    
    return predictions_dict

def create_prediction_plot(data, predictions, ticker, future_days):
    """Crée un graphique de prédiction"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Données historiques
    historical_data = data[ticker].iloc[-60:]
    ax.plot(historical_data.index, historical_data.values, 'b-', 
            label='Historique', linewidth=2, alpha=0.8)
    
    # Prédictions
    future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='D')[1:]
    ax.plot(future_dates, predictions[ticker], 'r--', 
            label=f'Prédiction ({future_days}j)', linewidth=2)
    
    # Ligne de connexion
    ax.plot([data.index[-1], future_dates[0]],
            [data[ticker].iloc[-1], predictions[ticker][0]],
            'r--', alpha=0.5)
    
    ax.set_title(f'{ticker} - Prédiction des Prix', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Informations de prédiction
    current_price = data[ticker].iloc[-1]
    predicted_price = predictions[ticker][-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
    ax.text(0.02, 0.98,
            f'Prix actuel: ${current_price:.2f}\n'
            f'Prédiction: ${predicted_price:.2f}\n'
            f'Variation: {change_pct:+.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=bbox_props, fontsize=10)
    
    return fig, current_price, predicted_price, change_pct

def create_text_report(data, tickers, predictions, optimal_weights, optimal_results, volatility):
    """Crée un rapport textuel détaillé"""
    report = "# RAPPORT D'ANALYSE FINANCIÈRE\n\n"
    report += f"## Données de Base\n"
    report += f"- Période: {data.index[0].date()} to {data.index[-1].date()}\n"
    report += f"- Nombre de jours: {len(data)}\n"
    report += f"- Actions analysées: {', '.join(tickers)}\n\n"
    
    report += "## Performances Détaillées\n"
    for ticker in tickers:
        initial_price = data[ticker].iloc[0]
        final_price = data[ticker].iloc[-1]
        total_return = ((final_price / initial_price) - 1) * 100
        predicted_price = predictions[ticker][-1]
        future_return = ((predicted_price / final_price) - 1) * 100
        
        report += f"### {ticker}\n"
        report += f"- Prix initial: ${initial_price:.2f}\n"
        report += f"- Prix final: ${final_price:.2f}\n"
        report += f"- Rendement période: {total_return:+.2f}%\n"
        report += f"- Prévision 30j: ${predicted_price:.2f}\n"
        report += f"- Variation attendue: {future_return:+.2f}%\n"
        report += f"- Volatilité: {volatility[ticker]:.4f}\n\n"
    
    if len(tickers) > 1:
        report += "## Optimisation du Portefeuille\n"
        report += "### Répartition Optimale\n"
        for i, ticker in enumerate(tickers):
            weight = optimal_weights[i] if i < len(optimal_weights) else 0
            if weight > 0.01:
                report += f"- {ticker}: {weight:.2%}\n"
        
        report += f"\n### Performance Attendue\n"
        report += f"- Rendement annuel: {optimal_results[0]:.2%}\n"
        report += f"- Volatilité annuelle: {optimal_results[1]:.2%}\n"
        report += f"- Ratio de Sharpe: {optimal_results[2]:.2f}\n"
    
    return report

# Interface principale
if analyze_btn and tickers_input:
    
    # Initialisation de la barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Étape 1: Téléchargement des données
    status_text.text("📥 Téléchargement des données...")
    data, tickers = download_yahoo_data(tickers_input, start_date, end_date)
    progress_bar.progress(25)
    
    if data is not None and len(tickers) > 0:
        st.success(f"✅ Données téléchargées pour {len(tickers)} action(s): {', '.join(tickers)}")
        
        # Aperçu des données
        with st.expander("📋 Aperçu des données"):
            st.write("**5 premières lignes:**")
            st.dataframe(data.head())
            st.write("**5 dernières lignes:**")
            st.dataframe(data.tail())
            st.write(f"**Statistiques descriptives:**")
            st.dataframe(data.describe())
        
        # Étape 2: Calcul des statistiques
        status_text.text("📊 Calcul des statistiques...")
        returns, mean_returns, volatility, correlation, cumulative_returns = calculate_statistics(data)
        progress_bar.progress(50)
        
        # Étape 3: Optimisation du portefeuille
        status_text.text("⚖️ Optimisation du portefeuille...")
        optimal_weights, optimal_results = optimize_portfolio(returns, tickers, num_portfolios)
        progress_bar.progress(75)
        
        # Étape 4: Prédictions IA
        status_text.text("🤖 Entraînement des modèles IA...")
        predictions = predict_prices(data, tickers, future_days)
        progress_bar.progress(100)
        status_text.text("✅ Analyse terminée!")
        
        # Affichage des résultats
        st.success("🎉 Analyse financière terminée avec succès!")
        
        # SECTION 1: VUE D'ENSEMBLE
        st.header("📊 Vue d'ensemble")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Actions analysées", len(tickers))
        with col2:
            total_return = ((data.iloc[-1] / data.iloc[0] - 1).mean() * 100)
            st.metric("Rendement moyen", f"{total_return:.1f}%")
        with col3:
            avg_volatility = volatility.mean() * 100
            st.metric("Volatilité moyenne", f"{avg_volatility:.2f}%")
        with col4:
            st.metric("Période analysée", f"{len(data)} jours")
        
        # SECTION 2: GRAPHIQUES PRINCIPAUX
        st.header("📈 Analyse Graphique")
        
        # Graphique 1: Prix historiques
        st.subheader("Évolution des Prix")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for ticker in tickers:
            ax1.plot(data.index, data[ticker], label=ticker, linewidth=2)
        ax1.set_ylabel('Prix ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Évolution des Prix', fontweight='bold')
        st.pyplot(fig1)
        plt.close(fig1)
        
        # Graphique 2: Rendements cumulés
        st.subheader("Rendements Cumulés")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for ticker in tickers:
            ax2.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker, linewidth=2)
        ax2.set_ylabel('Rendement Cumulé (1 = 100%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Rendements Cumulés', fontweight='bold')
        st.pyplot(fig2)
        plt.close(fig2)
        
        if len(tickers) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique 3: Matrice de corrélation
                st.subheader("Matrice de Corrélation")
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                im = ax3.imshow(correlation, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax3)
                ax3.set_xticks(range(len(tickers)))
                ax3.set_yticks(range(len(tickers)))
                ax3.set_xticklabels(tickers)
                ax3.set_yticklabels(tickers)
                ax3.set_title('Matrice de Corrélation', fontweight='bold')
                
                # Ajouter les valeurs
                for i in range(len(tickers)):
                    for j in range(len(tickers)):
                        ax3.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
                st.pyplot(fig3)
                plt.close(fig3)
            
            with col2:
                # Graphique 4: Volatilité
                st.subheader("Volatilité des Actions")
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                bars = ax4.bar(tickers, volatility.values, color=colors[:len(tickers)])
                ax4.set_ylabel('Volatilité Quotidienne')
                ax4.set_title('Volatilité des Actions', fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
                
                # Ajouter les valeurs
                for bar, value in zip(bars, volatility.values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
                st.pyplot(fig4)
                plt.close(fig4)
        else:
            st.info("⚠️ La matrice de corrélation nécessite au moins 2 actions")
        
        # SECTION 3: PRÉDICTIONS IA
        st.header("🔮 Prédictions IA")
        
        for ticker in tickers:
            st.subheader(f"Prédictions pour {ticker}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Graphique de prédiction
                fig_pred, current_price, predicted_price, change_pct = create_prediction_plot(
                    data, predictions, ticker, future_days
                )
                st.pyplot(fig_pred)
                plt.close(fig_pred)
            
            with col2:
                # Métriques de performance
                st.metric(
                    label="Prix Actuel",
                    value=f"${current_price:.2f}"
                )
                st.metric(
                    label=f"Prédiction {future_days}j",
                    value=f"${predicted_price:.2f}",
                    delta=f"{change_pct:+.1f}%"
                )
                
                # Calcul du rendement total
                initial_price = data[ticker].iloc[0]
                total_return = ((current_price / initial_price) - 1) * 100
                st.metric(
                    label="Rendement total",
                    value=f"{total_return:+.1f}%"
                )
        
        # SECTION 4: OPTIMISATION DU PORTEFEUILLE
        if len(tickers) > 1:
            st.header("⚖️ Optimisation du Portefeuille")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📋 Répartition Optimale")
                for i, ticker in enumerate(tickers):
                    weight = optimal_weights[i] if i < len(optimal_weights) else 0
                    if weight > 0.01:
                        st.markdown(f"**{ticker}**: {weight:.2%}")
            
            with col2:
                st.subheader("📊 Performance Attendue")
                st.markdown(f"**Rendement annuel**: {optimal_results[0]:.2%}")
                st.markdown(f"**Volatilité annuelle**: {optimal_results[1]:.2%}")
                st.markdown(f"**Ratio de Sharpe**: {optimal_results[2]:.2f}")
            
            with col3:
                st.subheader("💰 Recommandations")
                st.info("Portefeuille optimisé pour le meilleur ratio risque/rendement")
                for i, ticker in enumerate(tickers):
                    weight = optimal_weights[i] if i < len(optimal_weights) else 0
                    if weight > 0.1:
                        amount = weight * 10000
                        st.markdown(f"**{ticker}**: ${amount:.0f}")
        else:
            st.info("⚠️ L'optimisation de portefeuille nécessite au moins 2 actions")
        
        # SECTION 5: RAPPORT DÉTAILLÉ
        st.header("📄 Rapport Détaillé")
        
        with st.expander("Voir le rapport complet"):
            # Générer le rapport textuel
            text_report = create_text_report(data, tickers, predictions, optimal_weights, optimal_results, volatility)
            st.text_area("Rapport Complet", text_report, height=400)
        
        # Téléchargement du rapport textuel
        st.download_button(
            label="📥 Télécharger le rapport (TXT)",
            data=text_report,
            file_name="analyse_financiere.txt",
            mime="text/plain"
        )
    else:
        st.error("❌ Aucune donnée valide trouvée. Vérifiez les tickers et la période.")

else:
    # Page d'accueil
    st.markdown("""
    ## 🎯 Bienvenue dans l'Analyseur Financier IA
    
    Cette application utilise l'intelligence artificielle pour analyser les marchés financiers 
    et vous fournir des insights actionnables.
    
    ### 🚀 Fonctionnalités incluses:
    
    **📈 Analyse Technique Avancée**
    - Évolution des prix en temps réel
    - Calcul des rendements et volatilité
    - Matrices de corrélation
    
    **🤖 Prédictions par IA**
    - Modèles de machine learning
    - Prévisions à 30 jours
    - Analyse de tendances
    
    **⚖️ Optimisation de Portefeuille**
    - Théorie moderne du portefeuille
    - Maximisation du ratio de Sharpe
    - Recommandations d'allocation
    
    **📊 Rapports Complets**
    - Analyses détaillées
    - Visualisations interactives
    - Export des données
    
    ### 💡 Tickers d'exemple:
    - **Technologie**: AAPL, MSFT, GOOGL, TSLA, NVDA
    - **Finance**: JPM, BAC, GS
    - **Énergie**: XOM, CVX
    - **Consommation**: AMZN, WMT
    
    ### 📋 Comment utiliser:
    1. **Entrez les tickers** (séparés par des virgules)
    2. **Sélectionnez la période** d'analyse
    3. **Cliquez sur 'Lancer l'analyse'**
    4. **Explorez les résultats** dans les différentes sections
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📊 Application développée avec Streamlit | "
    "🤖 Powered by Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)