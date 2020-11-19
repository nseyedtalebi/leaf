"""if dp_params:
    # Optional params shared by both
    # batch_size = dp_params.get("batch_size", None)
    num_microbatches = dp_params.get("num_microbatches", None)
    unroll_microbatches = dp_params.get("unroll_microbatches", False)
    # DPGaussianOptimizerClass only
    l2_norm_clip = dp_params.get("l2_norm_clip", None)
    noise_multiplier = dp_params.get("noise_multiplier", None)
    dp_sum_query = dp_params.get("dp_sum_query", None)
    ledger = dp_params.get("ledger", None)
    

    is_dpgaussian = l2_norm_clip or noise_multiplier
    is_dp = bool(dp_sum_query)
    if is_dpgaussian:
        assert (
            l2_norm_clip and noise_multiplier
        ), "Must define l2_norm_clip and noise_multiplier to init DPGaussianOptimizer"
    else:
        assert (
            is_dp
        ), "Got non-empty dp_params that is missing required parameters. Must define dp_sum_query to init DPOptimizer or define l2_norm_clip and noise_multiplier to init DPGaussianOptimizer."
    if is_dpgaussian:
        _DPOptimizerCls = dp_optimizer.make_gaussian_optimizer_class(
            type(optimizer)
        )
    else:
        _DPOptimizerCls = dp_optimizer.make_optimizer_class(
            type(optimizer)
        )
    old_config = _optimizer.get_config()
    _optimizer = _DPOptimizerCls.from_config(
        {**old_config, **dp_params}
    )"""
