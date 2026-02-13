import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from schema import RAW_DATA_PATH, EDA_DATA_PATH

# Read data
df = pd.read_csv(RAW_DATA_PATH)

# 32 feat
pd.set_option('display.max_columns', 36)



def eda_process():
    global numerical_cols, categorical_cols
    # print(df.info())
    # print(df.nunique())

    # Numerical
    numerical_cols = df.select_dtypes(include=["number"]).columns
    # print(df[numerical_cols].describe().transpose())

    # Category
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns
    # print(df[categorical_cols].describe().transpose())

    # # Count fraud
    # fraud_vals = df['fraud_bool'].value_counts()
    # print(fraud_vals)

    # # Missing vals 
    df['missing_count'] = (
        (df == -1).sum(axis=1)
    )


def feature_engineering_datatype():
    global df
    
    # Unique number of categorical val small (2 ~ 7) 
    # --> One hot
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype="int8"
    )
    # print(df_encoded.head())
    
    # # Feat - target split
    # X = df_encoded.drop(['fraud_bool'], axis=1)
    # y = df_encoded['fraud_bool']
    
    # # Normalize numerical values
    # df = df.replace([np.inf, -np.inf], np.nan)
    # df[numerical_cols] = df[numerical_cols].fillna(0)

    # numerical_scaler = MinMaxScaler()
    # df[numerical_cols] = numerical_scaler.fit_transform(df[numerical_cols])


def feature_engineering_group():
    global df
    
    # 1. Application profile
    # Replace missing
    df['bank_months_count'] = df['bank_months_count'].replace(-1, 0)

    # High credit limit but low income --> sus
    df['credit_income_ratio'] = df['proposed_credit_limit'] / (df['income'] + 1)

    # High credit but new acc?
    df['credit_per_account_month'] = df['proposed_credit_limit'] / (df['bank_months_count'] + 1)

    # Younger/Older --> risk score low/high
    df['risk_age_ratio'] = df['credit_risk_score'] / (df['customer_age'] + 1)
    
    # 2. Application behaviour
    # High velo low session --> Bot
    df['velocity_per_session'] = df['velocity_6h'] / (df['session_length_in_minutes'] + 1)

    # 3. Contact info
    # Low name == email --> threshold (0.3?)
    df['low_name_email_similarity'] = (df['name_email_similarity'] < 0.3).astype(int)

    # Same dob but different devices? --> sus 
    df['cross_email_risk'] = df['date_of_birth_distinct_emails_4w'] * df['device_distinct_emails_8w']

    # 4. Velocity group (system behav)
    # Spike (?)
    df['velocity_spike'] = df['velocity_6h'] / (df['velocity_4w'] + 1)

    # From oversea?
    df['foreign_velocity'] = df['foreign_request'] * df['velocity_6h']

    # 5. Others
    # Many missing values + high credit risk --> fake profile?
    df['missing_credit_risk'] = df['missing_count'] * df['proposed_credit_limit']


def feature_selection():
    # Target split
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']
    
    # Handle inf 
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # 1. Variance threshold
    selector = VarianceThreshold()
    selector.fit(X)
    
    lowvar_features = [
        feature for feature in X.columns 
        if feature not in X.columns[selector.get_support()]
    ]
    
    # print(lowvar_features)
    X.drop(lowvar_features, axis=1, inplace=True)
    
    # 2. Check correlate (> 0.9)
    corr_matrix = X.corr().abs()

    upper_corr_matrix = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    correlated_cols = [
        column for column in upper_corr_matrix.columns 
        if any(upper_corr_matrix[column] > 0.9)
    ]

    # print(correlated_cols)
    
    # 3. Chi2 test 
    binary_feats = [
        col for col in X.columns
        if set(X[col].unique()).issubset({0,1})
    ]

    chi_scores, p_values = chi2(X[binary_feats], y)
    
    chi_df = pd.DataFrame({
        'feature': binary_feats,
        'chi2_score': chi_scores,
        'p_value': p_values
    }).sort_values(by='chi2_score', ascending=False)

    # print("Weak features:")
    # print(chi_df.tail(10))

    # 4. Mutual info
    numeric_feats = [
        col for col in X.columns
        if col not in binary_feats
    ]
    
    mi_scores = mutual_info_classif(X[numeric_feats], y)

    mi_df = pd.DataFrame({
        'feature': numeric_feats,
        'mi_score': mi_scores
    }).sort_values(by='mi_score', ascending=False)

    # print("Top MI features:")
    # print(mi_df.head(20))
    
    # 5. Extratrees importance
    extr_model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    
    extr_model.fit(X, y)
    
    importances = pd.Series(
        extr_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    # print("\nTop Tree Importance:")
    # print(importances.head(20))
    
    # Final select
    top_chi = set(chi_df.head(20)['feature'])
    top_mi = set(mi_df.head(20)['feature'])
    top_tree = set(importances.head(30).index)

    final_features = list(top_chi | top_mi | top_tree)

    X_selected = X[final_features]
    
    print(X_selected)

    df_final = pd.concat([X_selected, y], axis=1)

    df_final.to_csv(EDA_DATA_PATH, index=False)



def start_eda():
    eda_process()
    feature_engineering_datatype()
    feature_engineering_group()
    feature_selection()


start_eda()

# print(df.columns)
