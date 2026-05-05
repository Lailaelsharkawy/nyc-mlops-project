import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["processed_path"])

target = config["target"]

X = df.drop(columns=[target])
y = df[target]

num_cols = config["features"]["numeric"]
cat_cols = config["features"]["categorical"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("select", SelectKBest(score_func=f_regression, k=config["selection"]["k_best"]))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["split"]["test_size"],
    random_state=config["split"]["random_state"]
)

X_train = pipeline.fit_transform(X_train, y_train)
X_test = pipeline.transform(X_test)

joblib.dump(pipeline, "data/processed/preprocessor.pkl")

pd.DataFrame(X_train).to_csv("data/splits/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/splits/X_test.csv", index=False)

y_train.to_csv("data/splits/y_train.csv", index=False)
y_test.to_csv("data/splits/y_test.csv", index=False)

print("Preprocessing complete.")