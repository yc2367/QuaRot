import os, json
import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
                
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            assert ("llama" in args.model) or ("mistral" in args.model), "Only Llama and Mistral is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=args.cal_seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)


    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and (("llama" in args.model) or ("mistral" in args.model)):
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)

    if not args.lm_eval:
        for test_dataset in args.eval_dataset.split(','):
            # Evaluating on dataset
            testloader = data_utils.get_loaders(
                test_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                eval_mode=True
            )

            dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args, test_dataset)
            if args.wandb:
                wandb.log({'ppl/{}'.format(test_dataset.upper()): dataset_ppl})
            
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size, add_bos_token=False)

    if 'gsm8k_cot_llama' in args.tasks:
        num_fewshot = 8
        fewshot_as_multiturn = True
        apply_chat_template = True
    else:
        num_fewshot = 0
        fewshot_as_multiturn = False
        apply_chat_template = False
    
    confirm_run_unsafe_code = False
    for task in args.tasks:
        if 'humaneval' in task.lower():
            import os
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            confirm_run_unsafe_code = True
            break
    
    output_dir = os.path.join(f'/home/yc2367/llm/QuaRot/fake_quant/results/{args.tasks[0]}', args.model.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    if (args.w_bits == 16) and (args.a_bits == 16) and (args.k_bits == 16) and (args.v_bits == 16):
        output_file_name = "Baseline-FP16"
    else:
        output_file_name = f"w{args.w_bits}-a{args.a_bits}-gs{args.w_groupsize}-k{args.k_bits}-k{args.v_bits}"
    output_file_path = os.path.join(output_dir, f"{output_file_name}.json")
    if os.path.isfile(output_file_path):
        print(f'Found existing output file {output_file_name} for this experiment. Exit! \n\n')
        exit()

    results = lm_eval.simple_evaluate(
        hflm, 
        tasks=args.tasks, 
        batch_size=args.lm_eval_batch_size,
        num_fewshot=num_fewshot,
        fewshot_as_multiturn=fewshot_as_multiturn,
        apply_chat_template=apply_chat_template,
        confirm_run_unsafe_code=confirm_run_unsafe_code
    )

    # metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    # metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    # print(metric_vals)
    print(lm_eval_utils.make_table(results))
    print('\n')

    # Save results to JSON file
    with open(output_file_path, "w") as f:
        json.dump(results['results'], f, indent=4)

    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()
