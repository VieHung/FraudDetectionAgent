import argparse
from eda import start_eda
from train import Trainer
from schema import EDA_DATA_PATH, MODELS_DIR, TEST_PREDICT_INPUT_PATH

def main(mode, input_path, model_name):
    trainer = Trainer(data_path=EDA_DATA_PATH, model_dir=MODELS_DIR)

    if mode == "eda":
        start_eda()
    elif mode == "train":
        trainer.train_all()
    elif mode == "evaluate":
        trainer.evaluate_all(model_dir=MODELS_DIR)
    elif mode == "predict":
        if input_path is None:
            raise ValueError("Predict need input file")

        trainer.predict_batch(
            model_name=model_name,
            input_csv_path=input_path
        )
    else:
        print("Available mode: eda | train | evaluate | predict")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--input_path", type=str, default=TEST_PREDICT_INPUT_PATH)
    parser.add_argument("--model_name", type=str, default="lightgbm")

    args = parser.parse_args()

    main(args.mode, args.input_path, args.model_name)

# cli: python run.py --mode predict
