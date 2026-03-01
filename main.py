import torch
import numpy as np
import os
import re
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import AutoTokenizer, AutoModel, XLMRobertaModel, T5EncoderModel

from .processing.main import DataProcessor
from .model.train import MultiTaskModel
from .dataset.absa_module import ABSADatamodule
from .evaluation.eval import AspectBasedSentimentEvaluator
from .evaluation.metrics import evaluation_system_by_file
from .evaluation.filter_success import filter_exact_matches
from .evaluation.visual_aux import run_visual_demo
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'The model has {sum(p.numel() for p in model.parameters()):,} parameters')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataProcessor(
        train_txt_path=args.train_file,
        val_txt_path=args.val_file,
        test_txt_path=args.test_file,
        max_length=args.max_length,
    )
    dataframes = data.process(text_column="Review")
    train_df, val_df, test_df = dataframes["train"], dataframes["val"], dataframes["test"] 

    auxiliary_dict = {
        'AMBIENCE#GENERAL': "bầu không khí hoặc không gian của nhà hàng như thế nào?",
        'DRINKS#PRICES': "bạn đánh giá thế nào về giá của đồ uống?",
        'DRINKS#QUALITY': "chất lượng đồ uống ra sao?",
        'DRINKS#STYLE&OPTIONS': "lựa chọn và phong cách đồ uống như thế nào?",
        'FOOD#PRICES': "giá cả món ăn có hợp lý không?",
        'FOOD#QUALITY': "chất lượng món ăn được đánh giá như thế nào?",
        'FOOD#STYLE&OPTIONS': "phong cách và lựa chọn món ăn có đa dạng không?",
        'LOCATION#GENERAL': "vị trí của nhà hàng có thuận tiện không?",
        'RESTAURANT#GENERAL': "bạn cảm thấy thế nào về nhà hàng nói chung?",
        'RESTAURANT#MISCELLANEOUS': "có điều gì đặc biệt khác về nhà hàng không?",
        'RESTAURANT#PRICES': "mức giá chung của nhà hàng như thế nào?",
        'SERVICE#GENERAL': "chất lượng phục vụ của nhân viên ra sao?"
    }


    aspect_columns = [col for col in train_df.columns if col not in ["Review"]] 
    auxiliary_aspects = [auxiliary_dict[col] for col in aspect_columns]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "info" in args.model_name:
        encoder = XLMRobertaModel.from_pretrained(args.model_name, output_hidden_states=True)
    elif "t5" in args.model_name.lower():
        encoder = T5EncoderModel.from_pretrained(args.model_name, output_hidden_states=True)
    else:
        encoder = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    model = MultiTaskModel(
        num_aspects=len(train_df.columns) - 1,
        encoder=encoder,
        lambda_contrastive=args.lambda_contrastive,
        learning_rate=args.learning_rate,
        topk_layer=args.topk_layer,
        global_class_weights=[0.25,0.25,0.1,0.4] if "vlsp" not in args.model_name else [0.35,0.35,0.1,0.2]
    )
    model.to(device)
    count_parameters(model)

    datamodule = ABSADatamodule(train_df, val_df, test_df, auxiliary_aspects, encoder, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    datamodule.setup()

    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_epoch',
        save_top_k=args.save_top_k,
        mode='max', 
        filename='best_model',
        dirpath=args.output_dir
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        patience=args.patience,
        mode='max',
        verbose=True
    )
 
    logger = CSVLogger(
        save_dir=args.output_dir,
        name=''  
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_progress_bar=False,
        logger=logger
    )
 
    trainer.fit(model, datamodule)

    # Test the model
    checkpoint_path = checkpoint_callback.best_model_path
    # checkpoint_path = args.output_dir + "/best_model.ckpt"
    aspect_columns = [col for col in test_df.columns if col not in ["Review"]]
    evaluator = AspectBasedSentimentEvaluator(
        model_class=MultiTaskModel,
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        datamodule=datamodule,
        aspect_columns=aspect_columns,
        device=device
    )
    print("Evaluating model on test set...")

    test_output_path = args.output_dir + "/test_predictions.txt"
    gold_output_path = args.output_dir + "/test_gold.txt"
    global_output_path = args.output_dir + "/global_sentiment_predictions.txt"
    test_matched_path = args.output_dir + "/test_matched.txt"

    predict_file = evaluator.generate_prediction_file(test_output_path)
    gold_file = evaluator.generate_gold_file(gold_output_path)
    matched_count, compared_count, matched_examples = filter_exact_matches(predict_file, gold_file, test_matched_path)
    evaluator.evaluate_global_sentiment(output_path=global_output_path)

    print(f"Test predictions saved to {predict_file}")
    print(f"Test gold labels saved to {gold_file}")
    print(f"Global sentiment predictions saved to {global_output_path}")
    print(f"Exact match filtering: {matched_count} out of {compared_count} examples matched. Matched examples saved to {test_matched_path}")

    info = evaluation_system_by_file(gold_file, predict_file, eval_type='micro')
    # absa_results = evaluate_absa(
    #     gold_file=gold_file,
    #     pred_file=predict_file,
    #     aspect_columns=aspect_columns,
    #     average='macro'
    # )
    # info = info + "\n" + absa_results
    print(info)

    # save log to file
    info_path = os.path.join(args.output_dir, "log.txt")
    with open(info_path, 'w') as f:
        f.write(info)

    if args.visualize_aux:
        for _, example in enumerate(matched_examples):
            idx, sentence, labels_str = example.split("\n", 2)
            aspects = re.findall(r'\{([^,]+),', labels_str)
            os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
            try:
                run_visual_demo(
                    model=evaluator.model,
                    tokenizer=tokenizer,
                    aspect_keys=aspect_columns,
                    sentence=sentence,
                    aux_embs=datamodule.auxiliary_embeddings.to(device),
                    max_length=args.max_length,
                    save_heatmap=os.path.join(args.output_dir, "visualizations", f"visual_{idx}_{'_'.join(aspects)}.png"),
                    device=device
                )
            except Exception as e:
                print(f"Error visualizing example {idx}: {e}")


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Train a multi-task aspect-based sentiment analysis model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation data file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-large", help="Pretrained model name")
    parser.add_argument("--topk_layer", type=int, default=4, help="Number of top layers to use from BERT")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for input text")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints and logs")
    parser.add_argument("--save_top_k", type=int, default=1, help="Number of best models to save based on validation accuracy")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping based on validation accuracy")
    parser.add_argument("--log_path", type=str, default="./output.log", help="Path to save training logs")
    parser.add_argument("--lambda_contrastive", type=float, default=0.2, help="Weight for contrastive loss")
    parser.add_argument("--visualize_aux", type=int, default=1, help="Whether to run the auxiliary embedding visualization demo")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
