import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

warnings.filterwarnings('ignore')


def load_dataset():
    """Load and validate pulsar dataset"""
    print("Loading pulsar dataset...")
    try:
        df = pd.read_csv('./input/pulsar.csv')
        df.columns = ["Mean_IP", "Std_IP", "Kurtosis_IP", "Skewness_IP",
                      "Mean_DM_SNR", "Std_DM_SNR", "Kurtosis_DM_SNR", "Skewness_DM_SNR", "label"]

        # Convert all feature columns to numeric, handling mixed types
        feature_columns = ["Mean_IP", "Std_IP", "Kurtosis_IP", "Skewness_IP",
                           "Mean_DM_SNR", "Std_DM_SNR", "Kurtosis_DM_SNR", "Skewness_DM_SNR"]

        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert label column to numeric
        df['label'] = pd.to_numeric(df['label'], errors='coerce')

        # Handle missing values after conversion
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_count = df[numeric_cols].isnull().sum().sum()
        if missing_count > 0:
            # Fill missing values with column means
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            print(f"Filled {missing_count} missing values in numeric columns")

        # Clean up label column - ensure it's binary (0 or 1)
        unique_labels = df['label'].unique()
        print(f"Original unique labels: {sorted(unique_labels)}")

        # Map non-standard labels to binary
        df['label'] = df['label'].apply(lambda x: 1 if x > 0.5 else 0)

        print(f"Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
        print(f"Data types: {dict(df.dtypes)}")
        return df
    except FileNotFoundError:
        print("Error: Could not find './input/pulsar.csv'")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()


def create_sample_dataset():
    """Create a sample pulsar dataset for demonstration"""
    np.random.seed(42)
    n_samples = 2000

    # Generate features for non-pulsars (class 0) - majority class
    n_non_pulsar = int(n_samples * 0.9)
    non_pulsar_features = np.random.normal(0, 1, (n_non_pulsar, 8))
    non_pulsar_labels = np.zeros(n_non_pulsar)

    # Generate features for pulsars (class 1) - minority class
    n_pulsar = n_samples - n_non_pulsar
    pulsar_features = np.random.normal(2, 1.5, (n_pulsar, 8))  # Different distribution
    pulsar_labels = np.ones(n_pulsar)

    # Combine data
    X = np.vstack([non_pulsar_features, pulsar_features])
    y = np.hstack([non_pulsar_labels, pulsar_labels])

    # Create DataFrame
    feature_names = ["Mean_IP", "Std_IP", "Kurtosis_IP", "Skewness_IP",
                     "Mean_DM_SNR", "Std_DM_SNR", "Kurtosis_DM_SNR", "Skewness_DM_SNR"]

    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y.astype(int)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    print("Sample dataset created for demonstration")
    return df


def analysis(dataset):
    """Enhanced dataset analysis with visualizations"""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    # Basic info
    print(f"Dataset shape: {dataset.shape}")
    print(f"Missing values: {dataset.isnull().sum().sum()}")
    print(f"Data types:\n{dataset.dtypes}")

    # Class distribution
    vc = dataset['label'].value_counts().sort_index()
    print(f"\nCLASS DISTRIBUTION:")
    for label, count in vc.items():
        class_name = "Non-Pulsar" if label == 0 else "Pulsar"
        print(f"   {class_name} ({label}): {count:,} ({count / len(dataset) * 100:.1f}%)")

    # Statistical summary
    print(f"\nSTATISTICAL SUMMARY:")
    print(dataset.describe())

    # Create visualizations
    plt.style.use('default')
    sns.set_palette("husl")
    feature_cols = dataset.columns[:-1]

    # Feature distributions - only plot if all data is numeric
    try:
        # Verify all feature columns are numeric
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                numeric_features.append(col)
            else:
                print(f"Warning: Skipping non-numeric column '{col}' in visualization")

        if numeric_features:
            n_features = len(numeric_features)
            n_rows = (n_features + 3) // 4  # Calculate rows needed
            fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle('Feature Distributions by Class', fontsize=20, fontweight='bold')

            for i, col in enumerate(numeric_features):
                row, col_idx = i // 4, i % 4
                ax = axes[row, col_idx]
                for label in sorted(dataset['label'].unique()):
                    data = dataset[dataset['label'] == label][col]
                    # Remove any remaining non-numeric values
                    data = pd.to_numeric(data, errors='coerce').dropna()
                    if len(data) > 0:
                        class_name = "Non-Pulsar" if label == 0 else "Pulsar"
                        ax.hist(data, bins=30, alpha=0.7, label=class_name, density=True)
                ax.set_title(col, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for i in range(len(numeric_features), n_rows * 4):
                row, col_idx = i // 4, i % 4
                axes[row, col_idx].set_visible(False)

            plt.tight_layout()
            plt.savefig("plots/feature_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print("Saved feature distributions plot")
        else:
            print("Warning: No numeric features available for distribution plotting")
    except Exception as e:
        print(f"Error creating feature distributions: {str(e)}")

    # Correlation matrix - only for numeric columns
    try:
        numeric_df = dataset.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig("plots/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print("Saved correlation matrix plot")
        else:
            print("Warning: Not enough numeric columns for correlation matrix")
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")

    # Prepare data split - ensure all features are numeric
    feature_cols = dataset.columns[:-1]  # All columns except 'label'
    X = dataset[feature_cols].select_dtypes(include=[np.number])  # Only numeric columns
    y = dataset['label']

    print(f"Using {len(X.columns)} numeric features: {list(X.columns)}")

    if X.empty:
        raise ValueError("No numeric features available for modeling")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nDATA SPLIT: Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # Check class distribution in splits
    print(f"Train set class distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"Test set class distribution: {dict(y_test.value_counts().sort_index())}")

    return X_train, X_test, y_train, y_test


class SVMEvaluator:
    def __init__(self):
        self.results = []
        self.best_models = {}
        self.scaler = StandardScaler()
        plt.style.use('default')
        sns.set_palette("husl")

    def get_svm_configurations(self):
        """Define all SVM configurations with fixed parameters"""
        return {
            'RBF_Default': SVC(kernel='rbf', random_state=42),
            'Linear_Default': SVC(kernel='linear', random_state=42),
            'Poly_Degree2': SVC(kernel='poly', degree=2, random_state=42),
            'Poly_Degree3': SVC(kernel='poly', degree=3, random_state=42),
            'Sigmoid': SVC(kernel='sigmoid', random_state=42),
            'LinearSVC_Default': LinearSVC(random_state=42, max_iter=2000),
            'LinearSVC_L1': LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=2000),
            # Fixed NuSVC parameters - use smaller nu values
            'NuSVC_RBF': NuSVC(kernel='rbf', nu=0.1, random_state=42),
            'NuSVC_Linear': NuSVC(kernel='linear', nu=0.1, random_state=42),
            'RBF_Tuned': SVC(kernel='rbf', C=100, gamma='scale', random_state=42),
            'Linear_Tuned': SVC(kernel='linear', C=10, random_state=42),
            'Poly_Tuned': SVC(kernel='poly', degree=3, C=10, coef0=1, random_state=42),
        }

    def evaluate_single_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Comprehensive evaluation of a single model"""
        try:
            # Train and time
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                                       zero_division=0)

            # AUC score
            try:
                if hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                elif hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    y_scores = y_pred
                auc_score = roc_auc_score(y_test, y_scores)
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {model_name}: {str(e)}")
                auc_score = np.nan

            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            except Exception as e:
                print(f"Warning: Could not perform CV for {model_name}: {str(e)}")
                cv_mean = cv_std = np.nan

            result = {
                'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
                'F1-Score': f1, 'AUC': auc_score, 'CV_Mean': cv_mean, 'CV_Std': cv_std,
                'Training_Time': training_time, 'Model_Object': model, 'Predictions': y_pred
            }
            self.results.append(result)
            return result
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return None

    def hyperparameter_tuning(self, X_train, y_train):
        """Advanced hyperparameter tuning"""
        print("\nHYPERPARAMETER TUNING")
        tuning_configs = {
            'RBF_Optimized': {
                'model': SVC(kernel='rbf', random_state=42),
                'params': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]}
            },
            'Linear_Optimized': {
                'model': SVC(kernel='linear', random_state=42),
                'params': {'C': [0.01, 0.1, 1, 10, 100]}
            },
            'Poly_Optimized': {
                'model': SVC(kernel='poly', random_state=42),
                'params': {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}
            },
            'LinearSVC_Optimized': {
                'model': LinearSVC(random_state=42, max_iter=2000),
                'params': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}  # Removed l1 to avoid dual issue
            }
        }

        for name, config in tuning_configs.items():
            print(f"Tuning {name}...", end=" ")
            try:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=3,  # Reduced CV folds for faster execution
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                self.best_models[name] = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}, Score: {grid_search.best_score_:.4f}")
            except Exception as e:
                print(f"Error: {str(e)}")

    def create_comprehensive_visualizations(self, X_test, y_test):
        """Create comprehensive visualizations"""
        print("\nCREATING VISUALIZATIONS")

        if not self.results:
            print("No results to visualize")
            return

        results_df = pd.DataFrame(self.results)

        # Performance dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SVM Performance Dashboard', fontsize=20, fontweight='bold')

        # Accuracy
        sns.barplot(data=results_df, y='Model', x='Accuracy', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Accuracy Scores', fontweight='bold')
        for i, v in enumerate(results_df['Accuracy']):
            axes[0, 0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

        # F1-Score
        sns.barplot(data=results_df, y='Model', x='F1-Score', ax=axes[0, 1], palette='plasma')
        axes[0, 1].set_title('F1-Score', fontweight='bold')

        # AUC
        auc_data = results_df.dropna(subset=['AUC'])
        if not auc_data.empty:
            sns.barplot(data=auc_data, y='Model', x='AUC', ax=axes[0, 2], palette='coolwarm')
            axes[0, 2].set_title('AUC Scores', fontweight='bold')
        else:
            axes[0, 2].text(0.5, 0.5, 'No AUC data available', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('AUC Scores', fontweight='bold')

        # Training time
        sns.barplot(data=results_df, y='Model', x='Training_Time', ax=axes[1, 0], palette='magma')
        axes[1, 0].set_title('Training Time (seconds)', fontweight='bold')

        # CV scores
        cv_data = results_df.dropna(subset=['CV_Mean'])
        if not cv_data.empty:
            sns.barplot(data=cv_data, y='Model', x='CV_Mean', ax=axes[1, 1], palette='Set2')
            axes[1, 1].set_title('Cross-Validation', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No CV data available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Cross-Validation', fontweight='bold')

        # Performance vs Time
        sns.scatterplot(data=results_df, x='Training_Time', y='Accuracy', size='F1-Score',
                        hue='Model', ax=axes[1, 2], s=100)
        axes[1, 2].set_title('Performance vs Speed', fontweight='bold')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('plots/svm_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Saved performance dashboard")

        # Metrics heatmap
        plt.figure(figsize=(12, 10))
        heatmap_data = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].fillna(0)
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f')
        plt.title('SVM Performance Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/svm_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Saved metrics heatmap")

        # Top 4 confusion matrices
        top_models = results_df.nlargest(min(4, len(results_df)), 'Accuracy')
        n_models = len(top_models)
        n_rows = (n_models + 1) // 2
        n_cols = min(2, n_models)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle(f'Confusion Matrices - Top {n_models} Models', fontsize=18, fontweight='bold')

        for idx, (_, model_info) in enumerate(top_models.iterrows()):
            if n_models > 1:
                ax = axes[idx]
            else:
                ax = axes[0]
            cm = confusion_matrix(y_test, model_info['Predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{model_info['Model']}\nAccuracy: {model_info['Accuracy']:.3f}",
                         fontweight='bold')

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig('plots/top_models_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Saved confusion matrices")

        # ROC curves
        plt.figure(figsize=(12, 8))
        roc_plotted = False
        for _, model_info in results_df.dropna(subset=['AUC']).iterrows():
            try:
                model = model_info['Model_Object']
                if hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                elif hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    continue

                fpr, tpr, _ = roc_curve(y_test, y_scores)
                plt.plot(fpr, tpr, linewidth=2, label=f"{model_info['Model']} (AUC: {model_info['AUC']:.3f})")
                roc_plotted = True
            except Exception as e:
                print(f"Warning: Could not plot ROC for {model_info['Model']}: {str(e)}")
                continue

        if roc_plotted:
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('plots/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print("Saved ROC curves")
        else:
            print("No ROC curves could be plotted")

    def generate_detailed_report(self):
        """Generate comprehensive report"""
        if not self.results:
            print("No results to report")
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SVM EVALUATION REPORT")
        print("=" * 80)

        print(f"\nSUMMARY:")
        print(f"Models evaluated: {len(results_df)}")
        print(f"Best accuracy: {results_df['Accuracy'].max():.4f}")
        print(f"Average accuracy: {results_df['Accuracy'].mean():.4f}")
        print(f"Std accuracy: {results_df['Accuracy'].std():.4f}")

        print(f"\nTOP 5 PERFORMERS:")
        top_5 = results_df.nlargest(min(5, len(results_df)), 'Accuracy')
        for idx, (_, model) in enumerate(top_5.iterrows(), 1):
            print(f"{idx}. {model['Model']}: {model['Accuracy']:.4f} (F1: {model['F1-Score']:.4f})")

        # Kernel analysis
        kernel_performance = {}
        for _, model in results_df.iterrows():
            model_name = model['Model']
            if 'RBF' in model_name:
                kernel = 'RBF'
            elif 'Linear' in model_name:
                kernel = 'Linear'
            elif 'Poly' in model_name:
                kernel = 'Polynomial'
            elif 'Sigmoid' in model_name:
                kernel = 'Sigmoid'
            elif 'Nu' in model_name:
                kernel = 'Nu-SVM'
            else:
                kernel = 'Other'

            kernel_performance.setdefault(kernel, []).append(model['Accuracy'])

        print(f"\nKERNEL ANALYSIS:")
        for kernel, accs in kernel_performance.items():
            print(f"  {kernel}: {np.mean(accs):.4f} ± {np.std(accs):.4f} (n={len(accs)})")

        best_overall = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f"\nBEST MODEL: {best_overall['Model']}")
        print(f"   Accuracy: {best_overall['Accuracy']:.4f}")
        print(f"   F1-Score: {best_overall['F1-Score']:.4f}")
        print(f"   Training Time: {best_overall['Training_Time']:.2f}s")

        return results_df


def main():
    print("COMPREHENSIVE SVM ANALYSIS FOR PULSAR DETECTION")
    print("=" * 60)

    # Create output directory
    os.makedirs('plots', exist_ok=True)

    # Initialize evaluator
    evaluator = SVMEvaluator()

    try:
        # Load and analyze data
        df = load_dataset()
        X_train, X_test, y_train, y_test = analysis(df)

        # Scale features
        print("\nSCALING FEATURES...")
        X_train_scaled = evaluator.scaler.fit_transform(X_train)
        X_test_scaled = evaluator.scaler.transform(X_test)
        print("Features scaled using StandardScaler")

        # Evaluate all models
        svm_configs = evaluator.get_svm_configurations()
        print(f"\nEVALUATING {len(svm_configs)} SVM CONFIGURATIONS")
        print("-" * 50)

        for idx, (name, model) in enumerate(svm_configs.items(), 1):
            print(f"[{idx:2d}/{len(svm_configs)}] {name}...", end=" ")
            result = evaluator.evaluate_single_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
            if result:
                print(f"Acc: {result['Accuracy']:.4f}, F1: {result['F1-Score']:.4f}")
            else:
                print("Failed")

        # Hyperparameter tuning
        evaluator.hyperparameter_tuning(X_train_scaled, y_train)

        # Evaluate tuned models
        print(f"\nEVALUATING OPTIMIZED MODELS")
        print("-" * 40)
        for name, model in evaluator.best_models.items():
            print(f"Evaluating {name}...", end=" ")
            result = evaluator.evaluate_single_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
            if result:
                print(f"Acc: {result['Accuracy']:.4f}, F1: {result['F1-Score']:.4f}")
            else:
                print("Failed")

        # Create visualizations and report
        evaluator.create_comprehensive_visualizations(X_test_scaled, y_test)
        results_df = evaluator.generate_detailed_report()

        # Save results
        if not results_df.empty:
            output_df = results_df.drop(['Model_Object', 'Predictions'], axis=1, errors='ignore')
            output_df.to_csv('svm_evaluation_results.csv', index=False)
            print(f"\nResults saved to 'svm_evaluation_results.csv'")
            print(f"Visualization plots saved in 'plots/' directory")

        return evaluator, results_df

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    evaluator, results = main()
    if results is not None and not results.empty:
        print(f"\nAnalysis completed successfully!")
        print(f"Best performing model: {results.loc[results['Accuracy'].idxmax(), 'Model']}")
    else:
        print("\nAnalysis completed with issues. Check the output above for details.")