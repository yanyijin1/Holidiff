DISPLAY_NAME = {
    'HATEK': 'HoliDiff',
    'TSDiff': 'TSDiff',
    'SimDiff': 'SimDiff',
    'DiffusionTS': 'DiffusionTS',
    'FEDformer': 'FEDformer',
    'TimesNet': 'TimesNet',
    'TimeMixer': 'TimeMixer',
    'iTransformer': 'iTransformer',
    'PatchTST': 'PatchTST',
    'DLinear': 'DLinear',
    'CSDI': 'CSDI',
    'TimeGrad': 'TimeGrad',
}


def get_display_name(model_name: str) -> str:
    return DISPLAY_NAME.get(str(model_name), str(model_name))
