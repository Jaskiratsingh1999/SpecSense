import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

def load_data():
    data_path = r"C:\COMP4949_WorkSpace\Jaskirat-Assignment2\project\frontend\data\train.csv"
    return pd.read_csv(data_path)


def cross_fold_validation(x, y, model_type, test_size=0.2, n_splits=10):
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size)
    # Perform cross-validation on train+validation set
    k_fold = KFold(n_splits=n_splits, shuffle=True)
    rmse_list = []
    accuracy_list = []
    models = []
    print("\n===== Cross Fold Validation =====")
    for fold_idx, (train_index, val_index) in enumerate(k_fold.split(x_train_val)):
        print(f"\n----- Fold {fold_idx + 1}/{n_splits} -----")
        x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
        if model_type == "ols":
            model = sm.OLS(y_train, x_train).fit()
            val_predictions = model.predict(x_val)
            print("OLS Model Summary (Training):")
            print(model.summary())  # Only print coefficients table for brevity
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print("Validation RMSE:", rmse)
            rmse_list.append(rmse)
            accuracy = explained_variance_score(y_val, val_predictions)
            print("Validation Explained Variance:", accuracy)
            accuracy_list.append(accuracy)
            models.append(model)
        elif model_type == "bagging_regressor":
            model = BaggingRegressor(
                estimator=LinearRegression(),
                max_samples=0.5,
                max_features=1,
                n_estimators=100,
            )
            model.fit(x_train, y_train)
            val_predictions = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print("Validation RMSE:", rmse)
            rmse_list.append(rmse)
            accuracy = model.score(x_val, y_val)
            print("Validation R² Score:", accuracy)
            accuracy_list.append(accuracy)
            models.append(model)
        elif model_type == "ensemble_regressor":
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)
            svr = SVR(kernel="linear")
            lasso = Lasso()
            rf = RandomForestRegressor(n_estimators=5)
            individual_models = [svr, lasso, rf]
            print("Individual Model Performance (Validation):")
            for i, individual_model in enumerate(individual_models):
                individual_model.fit(x_train_scaled, y_train)
                val_predictions = individual_model.predict(x_val_scaled)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
                val_r2 = r2_score(y_val, val_predictions)
                print(f"  {individual_model.__class__.__name__} - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            stack = StackingCVRegressor(
                regressors=(svr, lasso, rf),
                meta_regressor=LinearRegression()
            )
            stack.fit(x_train_scaled, y_train)
            val_predictions = stack.predict(x_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print("Stacked Ensemble Validation RMSE:", rmse)
            rmse_list.append(rmse)
            accuracy = r2_score(y_val, val_predictions)
            print("Stacked Ensemble Validation R² Score:", accuracy)
            accuracy_list.append(accuracy)
            models.append((scaler, stack))
    print("\nCross-Validation Results:")
    print(f"Average RMSE: {np.mean(rmse_list):.4f} (±{np.std(rmse_list):.4f})")
    print(f"Average Accuracy: {np.mean(accuracy_list):.4f} (±{np.std(accuracy_list):.4f})")
    best_idx = np.argmin(rmse_list)
    best_model = models[best_idx]
    print("\n===== Final Test Set Evaluation =====")
    if model_type == "ols":
        test_predictions = best_model.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        test_exp_var = explained_variance_score(y_test, test_predictions)
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test Explained Variance: {test_exp_var:.4f}")
    elif model_type == "bagging_regressor":
        test_predictions = best_model.predict(x_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
    elif model_type == "ensemble_regressor":
        scaler, stack_model = best_model
        x_test_scaled = scaler.transform(x_test)
        test_predictions = stack_model.predict(x_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
    print("=================================\n")
    return best_model

def model_bagging_regressor(df):
    print("\n===== Bagging Regressor Model =====")
    x = df[["ram"]]
    x = sm.add_constant(x)
    y = df["price_range"]
    best_model = cross_fold_validation(x, y, "bagging_regressor")
    print("==================================\n")
    return best_model

def model_ensemble_regressor(df):
    print("\n===== Ensemble Regressor Model =====")
    x = df[["battery_power", "px_height", "px_width", "ram"]]

    y = df["price_range"]
    best_model = cross_fold_validation(x, y, "ensemble_regressor")
    print("====================================\n")
    return best_model

def model_stacked_regressor(df):
    print("\n===== Stacked Regressor Model =====")
    x = df[["battery_power", "px_height", "px_width", "ram"]]

    y = df["price_range"]
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2)

    def get_unfit_models():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma="scale"))
        models.append(AdaBoostRegressor())
        models.append(DecisionTreeRegressor())
        models.append(RandomForestRegressor(n_estimators=200))
        models.append(BaggingRegressor(n_estimators=200))
        models.append(ExtraTreesRegressor(n_estimators=200))
        return models

    def evaluate_model(x_data, y_data, predictions, model, name=""):
        rmse = np.sqrt(mean_squared_error(y_data, predictions))
        r2 = r2_score(y_data, predictions)
        print(f"{name}Model: {model.__class__.__name__}")
        print(f"{name}RMSE: {rmse:.4f}")
        print(f"{name}R² Score: {r2:.4f}")
        return rmse, r2

    def fit_base_models(x_train, y_train, x_test, models):
        df_predictions = pd.DataFrame()
        for i in range(len(models)):
            models[i].fit(x_train, y_train)
            predictions = models[i].predict(x_test)
            df_predictions[i] = predictions
        return df_predictions, models

    def fit_stacked_model(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model

    k_fold = KFold(n_splits=10, shuffle=True)
    cv_rmse_list = []
    cv_r2_list = []
    best_rmse = float('inf')
    best_models = None
    best_stacked = None
    print("\nCross-Fold Validation:")
    for fold_idx, (train_index, val_index) in enumerate(k_fold.split(x_train_val)):
        print(f"\n----- Fold {fold_idx + 1}/10 -----")
        x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        unfit_models = get_unfit_models()
        print("\nBase Models Validation Performance:")
        df_val_meta, trained_models = fit_base_models(x_train_scaled, y_train, x_val_scaled, unfit_models)
        for i in range(len(trained_models)):
            val_predictions = trained_models[i].predict(x_val_scaled)
            evaluate_model(x_val_scaled, y_val, val_predictions, trained_models[i])
        stacked_model = fit_stacked_model(df_val_meta, y_val)
        print("\nStacked Model Validation Performance:")
        stacked_predictions = stacked_model.predict(df_val_meta)
        val_rmse, val_r2 = evaluate_model(df_val_meta, y_val, stacked_predictions, stacked_model)
        cv_rmse_list.append(val_rmse)
        cv_r2_list.append(val_r2)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_models = (trained_models, scaler)
            best_stacked = stacked_model

    print("\nCross-Validation Results:")
    print(f"Average RMSE: {np.mean(cv_rmse_list):.4f} (±{np.std(cv_rmse_list):.4f})")
    print(f"Average R² Score: {np.mean(cv_r2_list):.4f} (±{np.std(cv_r2_list):.4f})")

    print("\nFinal Test Set Evaluation:")
    trained_models, scaler = best_models
    x_test_scaled = scaler.transform(x_test)
    print("\nBase Models Test Performance:")
    df_test_meta = pd.DataFrame()
    for i in range(len(trained_models)):
        test_predictions = trained_models[i].predict(x_test_scaled)
        df_test_meta[i] = test_predictions
        evaluate_model(x_test_scaled, y_test, test_predictions, trained_models[i])
    print("\nStacked Model Test Performance:")
    test_stacked_predictions = best_stacked.predict(df_test_meta)
    test_rmse, test_r2 = evaluate_model(df_test_meta, y_test, test_stacked_predictions, best_stacked)
    print("===================================\n")
    return (best_models, best_stacked)

def model_ols(df):
    print("\n===== OLS Linear Regression Model =====")
    x = df[["battery_power", "px_height", "px_width", "ram"]]

    y = df["price_range"]
    best_model = cross_fold_validation(x, y, "ols")
    print("======================================\n")
    return best_model

import joblib  # Add this at the top if not already there
import os

def main():
    df = load_data()

    # Train the stacked regressor model
    (trained_models, stacked_model) = model_stacked_regressor(df)
    
    # Extract scaler from trained_models tuple
    base_models, scaler = trained_models

    # Create output folder if not exists
    os.makedirs("backend/model", exist_ok=True)

    # Save the scaler and model
    joblib.dump(scaler, "backend/model/scaler.pkl")
    joblib.dump(base_models, "backend/model/base_models.pkl")
    joblib.dump(stacked_model, "backend/model/stacked_model.pkl")

    print("✅ Saved stacked model and scaler in backend/model/")

if __name__ == "__main__":
    main()
