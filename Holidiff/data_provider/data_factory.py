from torch.utils.data import DataLoader

from Holidiff.data_provider.traffic_warehouse_loader import TrafficWarehouseCsvDataset


data_dict = {
    'fujian30': TrafficWarehouseCsvDataset,
    'traffic_warehouse': TrafficWarehouseCsvDataset,
}


def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(
            f"Unsupported dataset '{args.data}'. "
            f"Available datasets: {list(data_dict.keys())}"
        )

    Data = data_dict[args.data]
    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size

    data_set = Data(
        csv_path=args.root_path + '/' + args.data_path,
        adj_path=getattr(args, 'adj_path', args.root_path + '/adjacent_gantry.csv'),
        split=flag,
        mode=getattr(args, 'mode', 'standard'),
        input_len=args.seq_len,
        pred_len=args.pred_len,
        stride=getattr(args, 'stride', 1),
        scale=getattr(args, 'scale', True),
        target_col=getattr(args, 'target_col', 'traffic_flow'),
        time_col=getattr(args, 'time_col', 'time_slot'),
        node_col=getattr(args, 'node_col', 'station_index'),
        holiday_col=getattr(args, 'holiday_col', 'is_holiday'),
        zero_as_missing=getattr(args, 'zero_as_missing', False),
        extreme_filter_threshold=getattr(args, 'extreme_filter_threshold', None),
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
