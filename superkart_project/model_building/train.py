import joblib, mlflow, pandas as pd, xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("superkart-sales-forecast")

Xtrain = pd.read_csv("Xtrain.csv"); Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze(); ytest = pd.read_csv("ytest.csv").squeeze()

numeric = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
categorical = ["Product_Sugar_Content", "Product_Type", "Store_Id",
               "Store_Size", "Store_Location_City_Type", "Store_Type"]
preprocessor = make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(handle_unknown="ignore"), categorical))
pipeline = make_pipeline(preprocessor, xgb.XGBRegressor(random_state=42, n_jobs=-1))
param_grid = {"xgbregressor__n_estimators": [100, 200],
              "xgbregressor__max_depth": [3, 5],
              "xgbregressor__learning_rate": [0.05, 0.1]}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(Xtrain, ytrain)
best = grid.best_estimator_
train_pred, pred = best.predict(Xtrain), best.predict(Xtest)
print("Best params:", grid.best_params_)
print("Test RMSE:", mean_squared_error(ytest, pred) ** 0.5)
print("Test R2  :", r2_score(ytest, pred))

with mlflow.start_run():
    for i, params in enumerate(grid.cv_results_["params"]):
        with mlflow.start_run(nested=True, run_name=f"combo_{i}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse", -grid.cv_results_["mean_test_score"][i])
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("train_rmse", mean_squared_error(ytrain, train_pred) ** 0.5)
    mlflow.log_metric("test_rmse", mean_squared_error(ytest, pred) ** 0.5)
    mlflow.log_metric("test_mae", mean_absolute_error(ytest, pred))
    mlflow.log_metric("test_r2", r2_score(ytest, pred))

joblib.dump(best, "superkart_project/deployment/best_model.joblib")
print("Model saved.")
