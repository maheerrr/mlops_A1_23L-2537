import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main():
    data_path = os.path.join("data", "used_car_price_dataset.csv")
    df = pd.read_csv(data_path)
    print("Dataset loaded:", df.shape)

    df = df.dropna()
    target_col = "price_usd"  
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = 100
    learning_rate=0.05

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained. Test score:", model.score(X_test, y_test))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, os.path.join("model", "model_23L-2537.pkl"))
    print("Model saved to model/model_23L-2537.pkl")

if __name__ == "__main__":
    main()