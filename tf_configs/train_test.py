import opennmt
from opennmt.layers.position import SinusoidalPositionEncoder
model = opennmt.models.Transformer(
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1,
    attention_dropout=0.1,
    pre_norm=True,
    position_encoder_class=SinusoidalPositionEncoder,
   
)


config = {
    "model_dir": "data/SMILES/",
    "data": {
        "source_vocabulary": "data/SMILES/src-vocab.txt",
        "target_vocabulary": "data/SMILES/tgt-vocab.txt",
        "train_features_file": "data/SMILES/src-train.txt",
        "train_labels_file": "data/SMILES/tgt-train.txt",
        "eval_features_file": "data/SMILES/src-val.txt",
        "eval_labels_file": "data/SMILES/tgt-val.txt"
    },
    "train": {
        "batch_size": 4096,
        "batch_type": "tokens",
        "keep_checkpoint_max": 20,
        "max_step": 1500000,
        "average_last_checkpoints": 20
    },
    "params": {
        "optimizer": "Adam",
        "optimizer_params": {
            "beta_1": 0.9,
            "beta_2": 0.998
        },
        "learning_rate": 2.0,
        "dropout": 0.1,
        "decay_type": "NoamDecay",
        "decay_params": {
             "model_dim": 512,
             "warmup_steps": 8000
        },
       "start_decay_steps": 50000,
       "label_smoothing": 0.1
   }
}


runner = opennmt.Runner(model, config, auto_config=True)
#runner.train(num_devices=1, with_eval=True, checkpoint_path="model_dir/", continue_from_checkpoint=True)
runner.train(num_devices=1, with_eval=True)
runner.evaluate(features_file="data/src-test.txt", labels_file="data/tgt-test.txt", checkpoint_path="data/SMILES")
runner.infer(features_file="data/src-test.txt", predictions_file="inferences/inference.txt", checkpoint_path="model_dir/")
runner.score(features_file="data/tgt-test.txt", predictions_file="inferences/inference.txt", checkpoint_path="model_dir/", output_file="inferences/results.txt")





