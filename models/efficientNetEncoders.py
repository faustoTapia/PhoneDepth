from tensorflow.keras import Model
from .efficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from .efficientNet import EfficientNetB5, EfficientNetB6, EfficientNetB7

##### Encoders Section ######

def efficientNetB0_encoder_old(input_shape=(224,224,3), input_tensor=None, weights='imagenet', name="efficientNetB0_encoder"):
    efficient_backbone = EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',         # In    x In    x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                     'block1a_project_bn',     # In/2  x In/2  x 16
                     'block2a_project_bn',     # In/4  x In/4  x 24    
                     'block3a_project_bn',     # In/8  x In/8  x 40
                     'block5a_project_bn',     # In/16 x In/16 x 112
                     'block7a_project_conv',     # In/32 x In/32 x 320
                    ]

    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    
    return efficientnet_encoder

def efficientNetB0_encoder(input_shape=(224,224,3), input_tensor=None, weights='imagenet', name="efficientNetB0_encoder"):
    efficient_backbone = EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In    x In    x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                     'block1a_project_bn',     # In/2  x In/2  x 16
                     'block2b_add',            # In/4  x In/4  x 24    
                     'block3b_add',            # In/8  x In/8  x 40
                     'block5c_add',            # In/16 x In/16 x 112
                     'block7a_project_conv',   # In/32 x In/32 x 320
                    ]

    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)

    return efficientnet_encoder

# Original: (240,240,3)
def efficientNetB1_encoder(input_shape=(256,256,3), input_tensor=None, weights='imagenet', name="efficientNetB1_encoder"):
    efficient_backbone = EfficientNetB1(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1b_add',            # In/2 x In/2 x 16
                    'block2c_add',            # In/4 x In/4 x 24
                    'block3c_add',            # In/8 x In/8 x 40
                    'block5d_add',            # In/16 x In/16 x 112
                    'block7b_add',            # In/32 x In/32 x 320
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    
    return efficientnet_encoder

# Original: (260,260,3)
def efficientNetB2_encoder(input_shape=(256,256,3), input_tensor=None, weights='imagenet', name="efficientNetB2_encoder"):
    efficient_backbone = EfficientNetB2(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1b_add',            # In/2 x In/2 x 16
                    'block2c_add',            # In/4 x In/4 x 24
                    'block3c_add',            # In/8 x In/8 x 48
                    'block5d_add',            # In/16 x In/16 x 120
                    'block7b_add',            # In/32 x In/32 x 352
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    
    return efficientnet_encoder

# Original: (300,300,3)
def efficientNetB3_encoder(input_shape=(288,288,3), input_tensor=None, weights='imagenet', name="efficientNetB3_encoder"):
    efficient_backbone = EfficientNetB3(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1b_add',            # In/2 x In/2 x 24
                    'block2c_add',            # In/4 x In/4 x 32
                    'block3c_add',            # In/8 x In/8 x 48
                    'block5e_add',            # In/16 x In/16 x 136
                    'block7b_add',            # In/32 x In/32 x 384
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    
    return efficientnet_encoder

# Original: (380,380,3)
def efficientNetB4_encoder(input_shape=(384,384,3), input_tensor=None, weights='imagenet', trainable=True, name="efficientNetB4_encoder"):
    as_model = input_tensor is None

    efficient_backbone = EfficientNetB4(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor,)
    
    for layer in efficient_backbone.layers:
        layer.trainable = trainable

    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1b_add',            # In/2 x In/2 x 24
                    'block2d_add',            # In/4 x In/4 x 32
                    'block3d_add',            # In/8 x In/8 x 56
                    'block5f_add',            # In/16 x In/16 x 160
                    'block7b_add',            # In/32 x In/32 x 448
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    
    if as_model:
        efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    else:
        efficientnet_encoder = outputs

    return efficientnet_encoder

# Original: (456,456,3)
def efficientNetB5_encoder(input_shape=(448,448,3), input_tensor=None, weights='imagenet', name="efficientNetB5_encoder"):
    efficient_backbone = EfficientNetB5(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1c_add',            # In/2 x In/2 x 24
                    'block2e_add',            # In/4 x In/4 x 40
                    'block3e_add',            # In/8 x In/8 x 64
                    'block5g_add',            # In/16 x In/16 x 176
                    'block7c_add',            # In/32 x In/32 x 512
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)

    return efficientnet_encoder

# Closest input shape to make it %32: (528,528,3)
def efficientNetB6_encoder(input_shape=(512,512,3), input_tensor=None, weights='imagenet', name="efficientNetB6_encoder"):
    efficient_backbone = EfficientNetB6(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1c_add',            # In/2 x In/2 x 32
                    'block2f_add',            # In/4 x In/4 x 40
                    'block3f_add',            # In/8 x In/8 x 72
                    'block5h_add',            # In/16 x In/16 x 200
                    'block7c_add',            # In/32 x In/32 x 576
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)
    
    return efficientnet_encoder

# Closest input shape to make it %32: (608,608,3)
def efficientNetB7_encoder(input_shape=(600,600,3), input_tensor=None, weights='imagenet', name="efficientNetB7_encoder"):
    efficient_backbone = EfficientNetB7(input_shape=input_shape, include_top=False, weights=weights, input_tensor=input_tensor)
    output_layers = ['stem_conv_pad',          # In x In x 3   After rescaling and normalization. Take input to this layer as previous layer is named dynamically.
                    'block1d_add',            # In/2 x In/2 x 32
                    'block2g_add',            # In/4 x In/4 x 48
                    'block3f_add',            # In/8 x In/8 x 72
                    'block5j_add',            # In/16 x In/16 x 224
                    'block7d_add',            # In/32 x In/32 x 640
                   ]
    
    outputs = [efficient_backbone.get_layer(layer_name).output for layer_name in output_layers[1:]]
    outputs = [efficient_backbone.get_layer(output_layers[0]).input] + outputs
    efficientnet_encoder = Model(inputs=efficient_backbone.input, outputs=outputs, name=name)

    return efficientnet_encoder

# Wrapper to access any efficienet encoder
def efficientNet_encoder(version, input_shape, input_tensor=None, weights='imagenet', trainable=True):
    print("Encoder: EfficientNetB{}".format(version))
    if type(version) != int or version > 7 or version < 0:
        raise ValueError('Version must be integer in range 0-7 for the efficientnet version')

    if version == 0:
        encoder = efficientNetB0_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 1:
        encoder = efficientNetB1_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 2:
        encoder = efficientNetB2_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 3:
        encoder = efficientNetB3_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 4:
        encoder = efficientNetB4_encoder(input_shape, input_tensor=input_tensor, trainable=trainable, weights=weights)
    elif version == 5:
        encoder = efficientNetB5_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 6:
        encoder = efficientNetB6_encoder(input_shape, input_tensor=input_tensor, weights=weights)
    elif version == 7:
        encoder = efficientNetB7_encoder(input_shape, input_tensor=input_tensor, weights=weights)

    return encoder
