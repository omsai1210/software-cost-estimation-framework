import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
from ml_model_training import train_and_evaluate, cocomo_organic
from deep_learning_model import build_and_train_mlp
from sklearn.preprocessing import StandardScaler

def create_comparison_plot(model_metrics):
    """
    model_metrics: dict of {model_name: {'mae': val, 'rmse': val}}
    """
    metrics = ['MAE', 'RMSE']
    model_names = list(model_metrics.keys())
    
    # Sort models by MAE for better visualization (or keep fixed order)
    # Let's keep a logical order: COCOMO, DL, RF, Hybrid
    order = ['Traditional COCOMO', 'Deep Learning (alone)', 'Random Forest (alone)', 'Hybrid Ensemble']
    model_names = [m for m in order if m in model_metrics]

    x = np.arange(len(metrics))
    width = 0.2

    # Set dark theme styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    colors = ['#ff3366', '#ffcc00', '#00ffcc', '#3399ff']
    
    for i, name in enumerate(model_names):
        values = [model_metrics[name]['mae'], model_metrics[name]['rmse']]
        offset = (i - (len(model_names)-1)/2) * width
        rects = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)], edgecolor='white')
        
        # Add labels
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=9)

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error Value', color='white', fontsize=12)
    ax.set_title('Final Model Performance Comparison', color='white', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='white', fontsize=12)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')

    # Grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    fig.tight_layout()

    # Save the plot
    plt.savefig('comparison_graph.png', transparent=True)
    print("Final comparison graph saved as comparison_graph.png")

if __name__ == "__main__":
    data_file = "software_projects_data.csv"
    print("Running all models to gather metrics for final visualization...")
    
    # 1. Base Preprocessing
    X, y = preprocess_data(data_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Random Forest
    rf_model, rf_mae, rf_rmse = train_and_evaluate(data_file)
    rf_preds = rf_model.predict(X_test)
    
    # 3. Deep Learning (alone)
    dl_model, dl_mae, dl_rmse = build_and_train_mlp(data_file)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    dl_preds = dl_model.predict(X_test_scaled).flatten()
    
    # 4. Hybrid Ensemble
    hybrid_preds = (0.95 * rf_preds) + (0.05 * dl_preds)
    hybrid_mae = mean_absolute_error(y_test, hybrid_preds)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_preds))
    
    # 5. Traditional COCOMO (using calibrated simulation logic)
    # Re-calculating with the same random seed for consistency
    np.random.seed(42)
    scaling_factors = np.random.uniform(2, 10, size=len(X_test))
    kloc_simulated = X_test['Complexity_Score'] * scaling_factors
    cocomo_pred_pm = np.array([cocomo_organic(k) for k in kloc_simulated])
    cocomo_pred_hours = cocomo_pred_pm * 160
    cocomo_mae = mean_absolute_error(y_test, cocomo_pred_hours)
    cocomo_rmse = np.sqrt(mean_squared_error(y_test, cocomo_pred_hours))

    # Compile all metrics
    all_metrics = {
        'Traditional COCOMO': {'mae': cocomo_mae, 'rmse': cocomo_rmse},
        'Deep Learning (alone)': {'mae': dl_mae, 'rmse': dl_rmse},
        'Random Forest (alone)': {'mae': rf_mae, 'rmse': rf_rmse},
        'Hybrid Ensemble': {'mae': hybrid_mae, 'rmse': hybrid_rmse}
    }

    create_comparison_plot(all_metrics)
