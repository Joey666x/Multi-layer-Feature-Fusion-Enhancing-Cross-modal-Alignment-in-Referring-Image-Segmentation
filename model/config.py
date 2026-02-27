from sacred import Experiment

ex = Experiment("VL")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "itc": 0,
        "itm_itc": 0,
        "irtr_itm_itc": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "snli": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    # below params varies with the environment
    root_dir = "~/MT"
    data_root = f"{root_dir}/dataset/fine-tune"
    log_dir = f"{root_dir}/logs"
    output_dir = f"{root_dir}/checkpoints"
    load_path = ""
    num_gpus = 8
    num_nodes = 1
    num_workers = 8
    precision = 32
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    per_gpu_eval_batchsize = 0

    # Wandb Logger Setting
    exp_name = "MT"
    group_name = "exp/task"
    run_name = "finetune"

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    log_every_n_steps = 50

    # Experiment Setting
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 0
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'CLIP-ViT-B/32'
    hidden_size = 512
    num_heads = 8
    #num_layers = 6
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 10
    max_steps = -1
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # Debug
    debug_num = 0

    # METER Setting
    meter_fusion = False
    vit_remove_last = False

    # BT Setting
    model_type = "MT"  # "METER", "BT", "MT"
    vit_layernorm_shared = True
    vit_layernorm_init_from_vit = False
    task_head_layers = 2  # 1, 2
    head_hidden_scale = 1  # 1, 2, 3, 4
    per_gpu_eval_batchsize_text = 256
    per_gpu_eval_batchsize_image = 128
    per_gpu_eval_batchsize_fusion_text = 500
    k_test = 128  # 128, 256
    amp_flag = True
    task_threshold = 0  # the task will be executed if it > task_threshold
    nlvr2_drop_rate = 0.1

    ## contrastive setting
    temperature = 0.07
    contrastive_hidden_size = 256
    gather_all_inputs = False
    gather_global_negative = False
    gather_all_image_inputs = False  # if all image features cannot be gathered in one GPU, then gather all image inputs
    image_chunks = 1  # if k_test x image need too many memory, then split them into chunks to calculate rerank scores
    text_chunks = 1  # if k_test x text need too many memory, then split them into chunks to calculate rerank scores
    save_memory = False

    ## MT Setting (Simplified)
    manager_type = 'AAUE'  # 'SAUE', 'AAUE'
    managed_layers_text = 6  # default top-k uni-modal layers
    managed_layers_image = 6  # default top-k uni-modal layers
    manager_learnable = True  # True, False
    manager_weight_type = 'vector'  # 'scalar', 'vector'
    manager_softmax_temperature = 1  # [0.07, 0.15, 0.2, 0.5, 1, 1.5]
    manager_softmax_temperature_learnable = True



