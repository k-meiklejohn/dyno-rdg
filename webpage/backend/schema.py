PARAM_SCHEMA = {
    "log_reduction":  {"type": "float",   "default": 1.0},
    "drop_length": {"type": "float",   "default": 0.3},
    "height_scale":  {"type": "float", "default": 2.0},
    "transcript_length": {'type': 'int', 'default': 1000, 'min':1}
}

ROW_SCHEMA = {
    "pos": {"type": "int", "default": 1, 'min': 1, 'max_param':'transcript_length'},
    "type": {"type": "select", 'options':['init','stop','shift+1', 'shift-1', 'ires', '5cap'], "default": '5cap'},
    "prob": {"type": "float", "default": 1, 'min': 0, 'max':1, 'step':0.05},
    "drop_prob": {"type": "float", "default": 0, 'min': 0, 'max':1, 'step':0.05},
}
