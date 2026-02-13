import pandas as pd
from schema import RAW_DATA_PATH, TEST_PREDICT_INPUT_PATH

df = pd.read_csv(RAW_DATA_PATH)

df_0 = df[df["fraud_bool"] == 0].sample(n=10, random_state=42)
df_1 = df[df["fraud_bool"] == 1].sample(n=10, random_state=42)

df_test = pd.concat([df_0, df_1]).reset_index(drop=True)

df_test.drop(["fraud_bool"], axis=1, inplace=True)

df_test.to_csv(TEST_PREDICT_INPUT_PATH)
