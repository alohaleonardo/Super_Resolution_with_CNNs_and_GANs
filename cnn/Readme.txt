see error when loading checkpoint:
    go to utils.py line 136, change torch.load('params—xxxxx',map_location={'cuda:x':'cuda:y'}
    x, y depends on your current setting
    
    drrn_b1u5: 6
    drrn_u9: 