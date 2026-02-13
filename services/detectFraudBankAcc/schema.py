import os

DATASET_DIR = "dataset"
MODELS_DIR = "models"
TEST_PREDICT_DIR = "test"

RAW_DATA_PATH = os.path.join(DATASET_DIR, "baf_base.csv")
EDA_DATA_PATH = os.path.join(DATASET_DIR, "baf_eda.csv")
TEST_PREDICT_INPUT_PATH = os.path.join(TEST_PREDICT_DIR, "input.csv")
TEST_PREDICT_OUTPUT_PATH = os.path.join(TEST_PREDICT_DIR, "predictions.csv")


RAW_FEATURES=[
    'income',
    'name_email_similarity',
    'prev_address_months_count',
    'current_address_months_count',
    'customer_age',
    'days_since_request',
    'intended_balcon_amount',
    'payment_type',
    'zip_count_4w',
    'velocity_6h',
    'velocity_24h',
    'velocity_4w',
    'bank_branch_count_8w',
    'date_of_birth_distinct_emails_4w',
    'employment_status',
    'credit_risk_score',
    'email_is_free',
    'housing_status',
    'phone_home_valid',
    'phone_mobile_valid',
    'bank_months_count',
    'has_other_cards',
    'proposed_credit_limit',
    'foreign_request',
    'source',
    'session_length_in_minutes',
    'device_os',
    'keep_alive_session',
    'device_distinct_emails_8w',
    'device_fraud_count',
    'month',
    'fraud_bool',
]