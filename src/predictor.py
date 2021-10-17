# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import lightgbm as lgb_original
from optuna.integration import lightgbm as lgb

class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2017-12-31"
    # 評価期間開始日
    VAL_START = "2018-02-01"
    # 評価期間終了日
    VAL_END = "2018-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"

    # 訓練期間終了日
    TRAIN_END2 = "2020-11-30"
 
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END]
                _val_X = feats[cls.VAL_START : cls.VAL_END]
                _test_X = feats[cls.TEST_START :]

                _train_y = labels[: cls.TRAIN_END]
                _val_y = labels[cls.VAL_START : cls.VAL_END]
                _test_y = labels[cls.TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y
    
    @classmethod
    def get_features_and_label_forTrain(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X = []
        trains_y = []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END2]
                _train_y = labels[: cls.TRAIN_END2]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                trains_y.append(_train_y)

        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)

        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)


        return train_X, train_y
    
    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        stock_list = dfs["stock_list"].copy()
        stock_fin = dfs["stock_fin"].copy()
        stock_price =dfs["stock_price"].copy()

        ### fin
        fin = stock_fin[stock_fin['Local Code'] == code]
        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        fin = fin.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        #finのnp.float64のデータのみを取得
        assets = ["Result_FinancialStatement NetAssets", "Result_FinancialStatement TotalAssets"]
        sales_and_incomes = ["Result_FinancialStatement NetSales", "Result_FinancialStatement NetIncome",
             "Forecast_FinancialStatement NetSales", "Forecast_FinancialStatement NetIncome",
             "Result_FinancialStatement OperatingIncome", "Result_FinancialStatement OrdinaryIncome",
              "Forecast_FinancialStatement OperatingIncome", "Forecast_FinancialStatement OrdinaryIncome"]
        fin_cols = list(assets) + list(sales_and_incomes)
        fin = fin[fin_cols]

        ###stock
        #17sector取得
        sector = stock_list[stock_list["Local Code"] == code]["17 Sector(Code)"].values
        #発行株式数取得
        issued_num = stock_list[stock_list["Local Code"] == code]["IssuedShareEquityQuote IssuedShare"].values
        #17sector代入
        fin["sector"] = np.int(sector)
        #発行株式数代入
        fin["issued_num"] = np.int(issued_num)

        #純利益率
        fin["NetReturn"] = fin["Result_FinancialStatement NetIncome"]/fin["Result_FinancialStatement NetSales"]
        #営業利益率
        fin["OperatingReturn"] = fin["Result_FinancialStatement OperatingIncome"]/fin["Result_FinancialStatement NetSales"]
        #経常利益率
        fin["OrdinaryReturn"] = fin["Result_FinancialStatement OrdinaryIncome"]/fin["Result_FinancialStatement NetSales"]
        #予測純利益率
        fin["ForecastNetReturn"] = fin["Forecast_FinancialStatement NetIncome"]/fin["Forecast_FinancialStatement NetSales"]
        #予測営業利益率
        fin["ForecastOperatingReturn"] = fin["Forecast_FinancialStatement OperatingIncome"]/fin["Forecast_FinancialStatement NetSales"]
        #予測経常利益率
        fin["ForecastOrdinaryReturn"] = fin["Forecast_FinancialStatement OrdinaryIncome"]/fin["Forecast_FinancialStatement NetSales"]

        columns = ["NetReturn", "OperatingReturn", "OrdinaryReturn", "ForecastNetReturn", "ForecastOperatingReturn", "ForecastOrdinaryReturn"]

        for col in columns:
            fin[col] = fin[col].pct_change(1)

        #自己資本比率
        fin["equity_ratio"] = fin["Result_FinancialStatement NetAssets"]/fin["Result_FinancialStatement TotalAssets"]
        
        # 元データのカラムを削除
        fin = fin.drop(sales_and_incomes, axis=1)

        #欠損値処理
        fin = fin.fillna(0)

        ###price
        # 特定の銘柄コードのデータに絞る
        price = stock_price[stock_price["Local Code"] == code]
        # 終値のみに絞る
        price= price[["EndOfDayQuote ExchangeOfficialClose"]]
        # 特徴量の生成対象期間を指定
        price = price.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

        # 終値の20営業日リターン
        price["return_1month"] = price[["EndOfDayQuote ExchangeOfficialClose"]].pct_change(20)
        # 終値の40営業日リターン
        price["return_2month"] = price[["EndOfDayQuote ExchangeOfficialClose"]].pct_change(40)
        # 終値の60営業日リターン
        price["return_3month"] = price[["EndOfDayQuote ExchangeOfficialClose"]].pct_change(60)
        # 終値の20営業日ボラティリティ
        price["volatility_1month"] = (
        np.log(price[["EndOfDayQuote ExchangeOfficialClose"]]).diff().rolling(20).std()
        )
        # 終値の40営業日ボラティリティ
        price["volatility_2month"] = (
        np.log(price[["EndOfDayQuote ExchangeOfficialClose"]]).diff().rolling(40).std()
        )
        # 終値の60営業日ボラティリティ
        price["volatility_3month"] = (
        np.log(price[["EndOfDayQuote ExchangeOfficialClose"]]).diff().rolling(60).std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        price["MA_gap_1month"] = price[["EndOfDayQuote ExchangeOfficialClose"]] / (
        price[["EndOfDayQuote ExchangeOfficialClose"]].rolling(20).mean()
        )
        # 終値と40営業日の単純移動平均線の乖離
        price["MA_gap_2month"] = price[["EndOfDayQuote ExchangeOfficialClose"]] / (
        price[["EndOfDayQuote ExchangeOfficialClose"]].rolling(40).mean()
        )
        # 終値と60営業日の単純移動平均線の乖離
        price["MA_gap_3month"] = price[["EndOfDayQuote ExchangeOfficialClose"]] / (
        price[["EndOfDayQuote ExchangeOfficialClose"]].rolling(60).mean()
        )
  
        # 60営業日平均
        price["stock_price"] = price[["EndOfDayQuote ExchangeOfficialClose"]].rolling(60).mean()
        # 元データのカラムを削除
        price = price.drop(["EndOfDayQuote ExchangeOfficialClose"], axis=1)

        #欠損値処理
        price = price.fillna(0)

        #おおまかな手順の３つ目
        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        price = price.loc[price.index.isin(fin.index)]
        fin = fin.loc[fin.index.isin(price.index)]

        # データを結合
        feats = pd.concat([price, fin], axis=1).dropna()

        #時価総額
        feats["market_cap"] = feats["stock_price"]*feats["issued_num"]
        #PBR
        feats["pbr"] = feats["stock_price"]/(fin["Result_FinancialStatement NetAssets"]*1000000/fin["issued_num"])

        # 元データのカラムを削除
        feats = feats.drop(["stock_price", "issued_num"], axis=1)
  
        columns = ["Result_FinancialStatement NetAssets", "Result_FinancialStatement TotalAssets", "market_cap"]

        for num in range(17):
            dat = feats[feats['sector'] == num+1].copy()
            for col in columns:
              mean = dat[col].mean()
              std = dat[col].std()
              dat[col] = (dat[col]-mean)/std
            feats[feats['sector'] == num+1] = dat.copy()

        # 元データのカラムを削除
        feats = feats.drop(["sector"], axis=1)

        # 欠損値処理
        feats = feats.replace([np.inf, -np.inf], 0)
        feats = feats.fillna(0)

        # 銘柄コードを設定
        feats["code"] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X, column_group="all"):
        # 特徴量グループを定義
        price_cols = ["return_1month", "return_2month", "return_3month",
                  "volatility_1month", "volatility_2month", "volatility_3month",
                  "MA_gap_1month", "MA_gap_2month", "MA_gap_3month"]
        technical_cols = ["market_cap", "pbr","profit_margin", "equity_ratio",
                  "Result_FinancialStatement NetAssets", "Result_FinancialStatement TotalAssets",
                  "NetReturn", "OperatingReturn", "OrdinaryReturn", "ForecastNetReturn", "ForecastOperatingReturn", "ForecastOrdinaryReturn"]
        all_cols = [x for x in train_X.columns if x != "code"]
        columns = {
        "price":price_cols,
        "technical":technical_cols,
        "all": all_cols
        }
        return columns[column_group]

    @classmethod
    def create_model(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            Lightgbm
        """
        # 特徴量を取得
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )
        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(dfs, train_X)
        # モデル作成
        lgb_train = lgb.Dataset(train_X[feature_columns], label = train_y)
        lgb_eval = lgb.Dataset(val_X[feature_columns], label = val_y)

        params = {
        'objective': 'mean_squared_error',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        }

        best = lgb.train(
        params, 
        lgb_train,
        valid_sets=lgb_eval, 
        early_stopping_rounds = 100)

        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y = cls.get_features_and_label_forTrain(
            dfs, codes, feature, label
        )
        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(dfs, train_X)
        # モデル作成
        lgb_train = lgb.Dataset(train_X[feature_columns], label = train_y)

        model = lgb_original.train(best.params, lgb_train, num_boost_round=1000)

        return model

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        """
        Args:
            model (lightgbm): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            m = os.path.join(model_path, f"my_model_{label}.pkl")
            with open(m, "rb") as f:
                # pickle形式で保存されているモデルを読み込み
                cls.models[label] = pickle.load(f)

        return True

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(label)
            model = cls.create_model(cls.dfs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]

        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, feats)

        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            df[label] = cls.models[label].predict(feats[feature_columns])
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()
