from data_processing import load_subject, CACHE_DIR
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
# Load a random subject

random_subject_id = random.randint(1, 32)
if len(str(random_subject_id)) == 1:
    random_subject_id = int(f"0{random_subject_id}")

X, y, vids = load_subject(random_subject_id, CACHE_DIR)


# Initialize the spliter

kfold_spliter = GroupKFold(n_splits=5)


# Loop over the fold

results = {}
for i, (train_index, test_index) in enumerate(kfold_spliter.split(X, y, groups=vids)):

    custom_scaler = StandardScaler()
    regressor = RandomForestRegressor(random_state=42)

    custom_scaler = custom_scaler.fit(X[train_index])
    scaled_train_X = custom_scaler.transform(X[train_index])
    scaled_test_X = custom_scaler.transform(X[test_index])

    regressor = regressor.fit(scaled_train_X, y[train_index])
    
    # predict on the scale test set
    y_pred = regressor.predict(scaled_test_X)
    y_true = y[test_index]
    

    # EVALUate MAE / RMSE / R²
    mae_score = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse_score = root_mean_squared_error(y_true, y_pred, multioutput='raw_values')

    # Store the metrics
    results[i] = {
        "mae": [mae_score[0], mae_score[1]], 
        "r2": [r2[0], r2[1]], 
        "rmse": [rmse_score[0], rmse_score[1]]
        }

V_metrics = {"mae": 0, "r2": 0, "rmse": 0}
A_metrics = {"mae": 0, "r2": 0, "rmse": 0}

for k, v in results.items():
    V_metrics["mae"] += v["mae"][0]
    V_metrics["r2"] += v["r2"][0]
    V_metrics["rmse"] += v["rmse"][0]

    A_metrics["mae"] += v["mae"][1]
    A_metrics["r2"] += v["r2"][1]
    A_metrics["rmse"] += v["rmse"][1]


V_metrics = {k: v / 5 for k, v in V_metrics.items()}
A_metrics = {k: v / 5 for k, v in V_metrics.items()}

