# install virtual env and then activate virtual env
#source venv_path/bin/activate

# BERT-Defense evaluation script
# Important parameter description can be found by ``python xx.py -h''
export CUDA_VISIBLE_DEVICES=0
#nohup 
python3 -u bert_defense_eval.py \
--cache_dir='../models' \
--test_dir='../M4_data_processed/processed_arxiv_bloomz.jsonl' \
--prediction_output='./eval_results/EVAL_processed_arxiv_bloomz.jsonl' \
--output_dir='./ckpts' \
--logging_file='./logging/EVAL_processed_arxiv_bloomz.jsonl' \
--tensor_logging_dir='./tf_logs' \
--train_batch_size=2 \
--val_batch_size=32 \
--model_ckpt_path='../models/bert_defense/checkpoint-10000' \
--num_train_epochs=1 \
--save_steps=260000 #\
#>./logs/EVAL_processed_arxiv_bloomz.txt &

