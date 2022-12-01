import argparse
from loguru import logger
import sys
import pandas as pd
sys.path.append('/Data/textgen')
from textgen.t5 import T5Model
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='data/asp_data/train_t5.csv', type=str, help='Training data file')
    parser.add_argument('--eval_file', default='data/asp_data/val_t5.csv', type=str, help='eval data file')
    parser.add_argument('--model_type', default='t5', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='Langboat/mengzi-t5-base', type=str, help='Transformers model or path')
    # parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    # parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--do_train', default='True', help='Whether to run training.')
    parser.add_argument('--do_predict', default='False', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='asp/exp/mengzi_t5_zh/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=10, type=int, help='Max sequence length')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        logger.info('Loading data...')
        # train_data: Pandas DataFrame containing the 3 columns - `prefix`, `input_text`, `target_text`.
        #   - `prefix`: A string indicating the task to perform. (E.g. `"question"`, `"stsb"`)
        #   - `input_text`: The input text. `prefix` is prepended to form the full input. (<prefix>: <input_text>)
        #   - `target_text`: The target sequence
        train_df = pd.read_csv(args.train_file)
        eval_df = pd.read_csv(args.eval_file)

        train_df['target_text'] = train_df['target_text'].astype(str)
        eval_df['target_text'] = eval_df['target_text'].astype(str)

        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_generated_text": True,
            "evaluate_during_training": True,
            "evaluate_during_training_verbose": True,
            "evaluate_during_training_steps":10000,
            "use_multiprocessing": False,
            "save_best_model": True,
            "save_steps":10000,
            "output_dir": args.output_dir,
            "use_early_stopping": True,
        }
        # model_type: t5  model_name: Langboat/mengzi-t5-base
        model = T5Model(args.model_type, args.model_name, args=model_args)

        def count_matches(labels, preds):
            logger.debug(f"labels: {labels[:10]}")
            logger.debug(f"preds: {preds[:10]}")
            match = sum([1 if label == pred else 0 for label, pred in zip(labels, preds)])
            logger.debug(f"match: {match}")
            return match

        model.train_model(train_df, eval_data=eval_df, matches=count_matches)
        print(model.eval_model(eval_df, matches=count_matches))

    if args.do_predict:
        model = T5Model(args.model_type, args.output_dir)
        sentences = ["什么是ai", "你是什么类型的计算机", "你知道热力学吗"]
        print("inputs:", sentences)
        print("outputs:", model.predict(sentences))


if __name__ == '__main__':
    main()