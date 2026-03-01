
export PYTHONPATH="/home/fit02/dien-workspace/absa/ViGSA:$PYTHONPATH"
echo "Running training script..."

BS=32
export CUDA_VISIBLE_DEVICES=0
# microsoft/infoxlm-large
# vinai/phobert-large
# VietAI/vit5-large
# google/rembert
# FacebookAI/xlm-roberta-large
# microsoft/xlm-align-base
# FPTAI/vibert-base-cased

python3 -m ViGSA.main \
    --train_file "/home/fit02/dien_workspace/absa/dataset/VLSP-ABSA/1-VLSP2018-SA-Restaurant-train.txt" \
    --val_file "/home/fit02/dien_workspace/absa/dataset/VLSP-ABSA/2-VLSP2018-SA-Restaurant-dev.txt" \
    --test_file "/home/fit02/dien_workspace/absa/dataset/VLSP-ABSA/3-VLSP2018-SA-Restaurant-test.txt" \
    --model_name "microsoft/infoxlm-large" \
    --topk_layer 6 \
    --batch_size 16 \
    --epochs 40 \
    --learning_rate 2e-5 \
    --output_dir "infoXLML-aux-asl-global-0.1contrastive-Res-vlsp-top6_new" \
    --save_top_k 1 \
    --patience 5 \
    --max_length 512 \
    --lambda_contrastive 0.1