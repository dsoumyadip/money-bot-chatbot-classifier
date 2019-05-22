import argparse
import os

from trainer.utils.utils import process_data
from trainer.model.estimator_keras import train_and_evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directory paths
    parser.add_argument('--base_dir',
                        help='Root directory',
                        required=True
                        )
    parser.add_argument('--data_dir',
                        help='Relative path of data directory',
                        required=True
                        )
    parser.add_argument('--local_file_name',
                        help='train file name',
                        required=True
                        )
    parser.add_argument('--external_file_name',
                        help='validation file name',
                        required=True
                        )
    parser.add_argument('--resources_dir',
                        help='Resourced directory path',
                        required=True
                        )
    parser.add_argument(
                        '--output_dir',
                        help='Directory to write checkpoints and export models',
                        required=True
                    )

    # Embeddings
    parser.add_argument("--embedding_size",
                        type=int,
                        default=32,
                        help="Word embedding size. (For glove, use 50 | 100 | 200 | 300)"
                        )

    # Model structure
    parser.add_argument("--sequence_length",
                        type=int,
                        default=30,
                        help="LSTM network length"
                        )
    parser.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="LSTM network depth"
                        )

    # Train params
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-2,
                        help="Learning rate."
                    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size."
                    )
    parser.add_argument("--keep_prob",
                        type=float,
                        default=1,
                        help="Dropout keep prob."
                    )
    parser.add_argument(
                        '--train_steps',
                        help='Steps to run the training job for',
                        type=int,
                        default=300
                    )
    parser.add_argument(
                        '--train_test_ratio',
                        help='Train and test data splitting ratio',
                        default=1,
                        type=float
                    )
    parser.add_argument(
                        '--validation_split',
                        help='Validation done on percentage of train data ',
                        default=0.1,
                        type=float
                    )

    args = parser.parse_args()
    arguments = args.__dict__

    print(arguments)

    local_file_path = os.path.join(arguments['base_dir'], os.path.join(arguments['data_dir'], os.path.join(arguments['local_file_name'])))
    external_file_path = os.path.join(arguments['base_dir'], os.path.join(arguments['data_dir'], os.path.join(arguments['external_file_name'])))
    resources_dir = os.path.join(arguments['base_dir'], arguments['resources_dir'])

    df = process_data(local_file_path, external_file_path, resources_dir)
    train_and_evaluate(data=df, args=arguments)
