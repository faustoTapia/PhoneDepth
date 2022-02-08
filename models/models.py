from numpy.lib.financial import ipmt

from models.compileStrategies import compile_md, compile_md_doubleDepth
from .park_mai_depth import bts_model_f as park_mai_depth
from .fastDepth import FastDepth
from .effnet_parkmai import effiparkmai_net
from .effnet_parkmai import effi_dualatt_parkmai_net, effiDepth_dualatt_parkmai_net


def depth_model(model_string, compile_strategy=None):

    # Models for paper.
    if 'p_fastdepth' in model_string:
        model = FastDepth((224, 224))
        if compile_strategy == 'md':
            compile_md(model)
        elif compile_strategy == 'md_dd':
            compile_md_doubleDepth(model)
        
    elif 'p_parkmai' in model_string:
        model = park_mai_depth(compile_strategy=compile_strategy)
    elif 'p_effiB4park' in model_string and 'ID' in model_string:
        model = effiDepth_dualatt_parkmai_net(4, (384,384), n_dualatt=0, dec_n_groups=1, compile_strategy=compile_strategy)
    elif 'p_effiB4park' in model_string and 'ID' not in model_string:
        if 'md' in model_string and 'fineTune' not in model_string:
            model = effiparkmai_net(4, input_shape=(384, 384, 3), separated_submodules=True, compile_strategy=compile_strategy)
        else:
            model = effi_dualatt_parkmai_net(4, (384, 384, 3), n_dualatt=0, dec_n_groups=1, compile_strategy=compile_strategy)

    else:
        raise ValueError("Model {} not found".format(model_string))

    return  model
      
final_models_checkpoints = {

    # Trained on MAI and fine-tunned
    "p_fastdepth_mai_224x224":                                  118,    #   0.2764
    "p_parkmai_mai_224x224":                                      92,   #   0.304
    "p_effiB4park_mai_384x384":                                 106,    #   0.2461
    "p_fastdepth_mai_224x224_fineTuneMB":                       73,     #   0.2748
    "p_parkmai_mai_224x224_fineTuneMB":                         79,     #   0.2767
    "p_effiB4park_mai_384x384_fineTuneMB":                      106,    #   0.2227

    # Trained on MD and fine-tunned
    "p_fastdepth_md_224x224":                                   60,     #   0.09315
    "p_parkmai_md_224x224":                                     41,     #   0.09724
    "p_effiB4park_md_384x384":                                  59,     #   0.062
    "p_fastdepth_md_224x224_fineTuneMB":                        63,     #   0.09002
    "p_parkmai_md_224x224_fineTuneMB":                          61,     #   0.09497
    "p_effiB4park_md_384x384_fineTuneMB":                       63,     #   0.05997
      
    "p_fastdepth_mbI2P_224x224":                               111,     #   0.28
    "p_parkmai_mbI2P_224x224":                                 119,     #   0.2883
    "p_effiB4park_mbI2P_384x384":                              90,      #   0.2411
    "p_fastdepth_mbI2DP_224x224":                              86,      #   0.2706
    "p_parkmai_mbI2DP_224x224":                                118,     #   0.2674
    "p_effiB4park_mbI2DP_384x384":                             119,      #  0.2344

    # Depth enhancement.
    "p_effiB4park_mbID2P_384x384":                             91,     #   0.2219
    "p_effiB4park_mbID2DP_384x384":                            110,     #   0.221
}
