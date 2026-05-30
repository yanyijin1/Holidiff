def print_args(args):
    from Holidiff.utils.display_name import get_display_name

    def g(name, default=''):
        return getattr(args, name, default)

    network_profile = g('network_profile', '')
    if network_profile:
        network_profile = str(network_profile)
    else:
        network_profile = 'custom'
    print("\033[1m" + "Basic Config" + "\033[0m")
    display_model = get_display_name(g('model', ''))
    print(f'  {"Task Name:":<20}{g("task_name"):<20}{"Is Training:":<20}{g("is_training"):<20}')
    print(f'  {"Model ID:":<20}{g("model_id"):<20}{"Model:":<20}{display_model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{g("data"):<20}{"Root Path:":<20}{g("root_path"):<20}')
    print(f'  {"Data Path:":<20}{g("data_path"):<20}{"Features:":<20}{g("features"):<20}')
    print(f'  {"Target:":<20}{g("target"):<20}{"Freq:":<20}{g("freq"):<20}')
    print(f'  {"Checkpoints:":<20}{g("checkpoints"):<20}')
    print()

    if g('task_name') in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{g("seq_len"):<20}{"Label Len:":<20}{g("label_len"):<20}')
        print(f'  {"Pred Len:":<20}{g("pred_len"):<20}{"Seasonal Patterns:":<20}{g("seasonal_patterns"):<20}')
        print(f'  {"Inverse:":<20}{g("inverse"):<20}')
        print()

    if g('task_name') == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{g("mask_rate"):<20}')
        print()

    if g('task_name') == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{g("anomaly_ratio"):<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Network Profile:":<20}{network_profile:<20}{"Top k:":<20}{g("top_k", "-"):<20}')
    print(f'  {"Num Kernels:":<20}{g("num_kernels", "-"):<20}{"Enc In:":<20}{g("enc_in", "-"):<20}')
    print(f'  {"Dec In:":<20}{g("dec_in", "-"):<20}{"C Out:":<20}{g("c_out", "-"):<20}')
    print(f'  {"d model:":<20}{g("d_model", "-"):<20}{"n heads:":<20}{g("n_heads", "-"):<20}')
    print(f'  {"e layers:":<20}{g("e_layers", "-"):<20}{"d layers:":<20}{g("d_layers", "-"):<20}')
    print(f'  {"d FF:":<20}{g("d_ff", "-"):<20}{"Moving Avg:":<20}{g("moving_avg", "-"):<20}')
    print(f'  {"Factor:":<20}{g("factor", "-"):<20}{"Distil:":<20}{g("distil", "-"):<20}')
    print(f'  {"Dropout:":<20}{g("dropout", "-"):<20}{"Embed:":<20}{g("embed", "-"):<20}')
    print(f'  {"Activation:":<20}{g("activation", "-"):<20}')
    if hasattr(args, 'output_attention'):
        print(f'  {"Output Attention:":<20}{g("output_attention"):<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{g("num_workers", "-"):<20}{"Itr:":<20}{g("itr", "-"):<20}')
    print(f'  {"Train Epochs:":<20}{g("train_epochs", "-"):<20}{"Batch Size:":<20}{g("batch_size", "-"):<20}')
    print(f'  {"Patience:":<20}{g("patience", "-"):<20}{"Learning Rate:":<20}{g("learning_rate", "-"):<20}')
    print(f'  {"Des:":<20}{g("des", "-"):<20}{"Loss:":<20}{g("loss", "-"):<20}')
    print(f'  {"Lradj:":<20}{g("lradj", "-"):<20}{"Use Amp:":<20}{g("use_amp", "-"):<20}')
    print()

    if hasattr(args, 'physical_injection_enable') or hasattr(args, 'frequency_enable'):
        print("\033[1m" + "Physical / Frequency Modules" + "\033[0m")
        if hasattr(args, 'physical_injection_enable'):
            print(f'  {"Physical Injection:":<20}{g("physical_injection_enable", False):<20}{"Physical Residual:":<20}{g("physical_residual_enable", False):<20}')
            print(f'  {"Residual Eta:":<20}{g("physical_residual_eta", 0.0):<20}{"FreeFlow Eps:":<20}{g("physical_free_flow_epsilon", 0.0):<20}')
        if hasattr(args, 'frequency_enable'):
            print(f'  {"Frequency Enable:":<20}{g("frequency_enable", False):<20}{"Freq Decomp:":<20}{g("frequency_decomp_type", "fixed_fft"):<20}')
            print(f'  {"Freq Residual:":<20}{g("frequency_residual_type", "hybrid_residual"):<20}{"Num Bands:":<20}{g("frequency_num_bands", 0):<20}')
            print(f'  {"Freq Inject:":<20}{g("frequency_injection_mode", "embed_replace"):<20}{"Hybrid Mix:":<20}{g("frequency_hybrid_mix_alpha", 0.0):<20}')
            if hasattr(args, 'frequency_conditioner_mode'):
                print(f'  {"Freq Cond Mode:":<20}{g("frequency_conditioner_mode"):<20}{"Cond Dim:":<20}{g("frequency_conditioner_dim"):<20}')
        print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{g("use_gpu"):<20}{"GPU:":<20}{g("gpu"):<20}')
    print(f'  {"Use Multi GPU:":<20}{g("use_multi_gpu"):<20}{"Devices:":<20}{g("devices"):<20}')
    print()

    if hasattr(args, 'p_hidden_dims') or hasattr(args, 'p_hidden_layers'):
        print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
        p_hidden_dims = g('p_hidden_dims', [])
        p_hidden_dims_str = ', '.join(map(str, p_hidden_dims)) if isinstance(p_hidden_dims, (list, tuple)) else str(p_hidden_dims)
        print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{g("p_hidden_layers", "-"):<20}')
        print()
