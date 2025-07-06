tasks="arc_challenge_chat"

if [[ ${tasks} == "arc_challenge_chat" ]]
then
    batch_size=16
elif [[ ${tasks} == "gsm8k_cot_llama" ]]
then
    batch_size=4
elif [[ ${tasks} == "humaneval_instruct" ]]
then
    batch_size=1
fi

echo "Batch Size: ${batch_size}"
echo "Tasks: ${tasks}"


####################  Test FP16  ####################
model="meta-llama/Llama-3.2-3B-Instruct"
python main.py --model ${model}  \
    --a_bits 16 --k_bits 16 --v_bits 16 --w_bits 16 \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}

model="meta-llama/Llama-3.1-8B-Instruct"
python main.py --model ${model}  \
    --a_bits 16 --k_bits 16 --v_bits 16 --w_bits 16 \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}


####################  Llama-3.2-3B-Instruct  ####################
model="meta-llama/Llama-3.2-3B-Instruct"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}


####################  Llama-3.1-8B-Instruct  ####################
model="meta-llama/Llama-3.1-8B-Instruct"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --lm_eval --lm_eval_batch_size ${batch_size} --tasks ${tasks}

