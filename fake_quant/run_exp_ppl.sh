cd ..


####################  Llama-7B  ####################
model="huggyllama/Llama-7b"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"


####################  Llama-13B  ####################
model="huggyllama/Llama-13b"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"


####################  Llama-2-7B  ####################
model="meta-llama/Llama-2-7b-hf"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"


####################  Llama-2-13B  ####################
model="meta-llama/Llama-2-13b-hf"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"


####################  Llama-3.1-8B  ####################
model="meta-llama/Llama-3.1-8B"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4" --bsz 8

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4" --bsz 8


####################  Llama-3.2-3B  ####################
model="meta-llama/Llama-3.2-3B"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4" --bsz 8

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4" --bsz 8


####################  Mistral-7B  ####################
model="mistralai/Mistral-7B-v0.3"
python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize 128 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize 128 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"

python main.py --model ${model}  \
    --rotate --a_bits 8 --a_groupsize -1 --a_clip_ratio 0.9 \
    --k_bits 4 --k_groupsize 128 --k_clip_ratio 0.95 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_clip_ratio 0.95 --v_asym \
    --w_bits 4 --w_groupsize -1 --w_clip --w_asym \
    --cal_dataset "pile" --cal_seqlen 512 \
    --eval_dataset "wikitext2,c4"
    