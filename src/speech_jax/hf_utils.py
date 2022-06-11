def hf_save_fn(
    save_dir,
    params,
    model_save_fn,
    feature_extractor_save_fn,
    tokenizer_save_fn,
    push_to_hub=False,
):
    model_save_fn(save_dir, params=params, push_to_hub=push_to_hub)
    feature_extractor_save_fn(save_dir, push_to_hub=push_to_hub)
    tokenizer_save_fn(save_dir, push_to_hub=push_to_hub)
