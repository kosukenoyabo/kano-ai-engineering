import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
PERFORMANCE_METRICS_PATH = os.path.join(MODEL_DIR, ".performance_metrics.json")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def trained_model_and_test_data(sample_data, preprocessor):
    """モデルの学習とテストデータの準備、モデルの保存も行う"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists(trained_model_and_test_data):
    """モデルファイルが存在するか確認"""
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def get_model_accuracy(trained_model_and_test_data):
    """モデルの精度を計算して返す"""
    model, X_test, y_test = trained_model_and_test_data

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def get_model_inference_time(trained_model_and_test_data):
    """モデルの推論時間を計算して返す"""
    model, X_test, _ = trained_model_and_test_data

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time


def test_initial_model_performance(trained_model_and_test_data):
    """初期モデルの性能基準をテスト（精度と推論時間）"""
    accuracy = get_model_accuracy(trained_model_and_test_data)
    inference_time = get_model_inference_time(trained_model_and_test_data)

    print(f"Current Accuracy: {accuracy}")
    print(f"Current Inference Time: {inference_time}")

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"
    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_performance_regression(trained_model_and_test_data):
    """過去の性能と比較して劣化がないか検証し、今回の性能を保存する"""
    current_accuracy = get_model_accuracy(trained_model_and_test_data)
    current_inference_time = get_model_inference_time(trained_model_and_test_data)

    print(f"Current Accuracy for regression test: {current_accuracy}")
    print(f"Current Inference Time for regression test: {current_inference_time}")

    previous_metrics = {}
    if os.path.exists(PERFORMANCE_METRICS_PATH):
        with open(PERFORMANCE_METRICS_PATH, "r") as f:
            try:
                previous_metrics = json.load(f)
            except json.JSONDecodeError:
                pytest.skip("パフォーマンスメトリクスファイルが不正な形式です。初回実行として扱います。")

    if previous_metrics:
        prev_accuracy = previous_metrics.get("accuracy")
        prev_inference_time = previous_metrics.get("inference_time")

        if prev_accuracy is not None:
            # 精度が著しく低下していないか (例: 前回比90%未満になったらエラー)
            assert current_accuracy >= prev_accuracy * 0.9, \
                f"精度が前回({prev_accuracy:.4f})から著しく低下しました({current_accuracy:.4f})"
        else:
            print("前回の精度データがありません。")

        if prev_inference_time is not None:
            # 推論時間が著しく増加していないか (例: 前回比150%を超えたらエラー)
            assert current_inference_time <= prev_inference_time * 1.5, \
                f"推論時間が前回({prev_inference_time:.4f}s)から著しく増加しました({current_inference_time:.4f}s)"
        else:
            print("前回の推論時間データがありません。")
    else:
        print("前回のパフォーマンスメトリクスが見つかりません。初回実行として扱います。")

    # 今回の性能を保存
    current_metrics_to_save = {
        "accuracy": current_accuracy,
        "inference_time": current_inference_time,
        "timestamp": time.time()
    }
    with open(PERFORMANCE_METRICS_PATH, "w") as f:
        json.dump(current_metrics_to_save, f, indent=4)
    print(f"現在のパフォーマンスメトリクスを保存しました: {PERFORMANCE_METRICS_PATH}")


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"
