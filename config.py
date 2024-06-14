from model.models import THGNN, TGCN 

seed = 0
timestep_hidden = 5
batch_size = 1

model_dict = {
    'THGNN': {
        'class': THGNN,
        'default_args': {
            'struc': 'hgraph',
            'seed': seed,
            'lr': 0.001,
            'epoch': 100,
            'timestep_hidden': timestep_hidden,
            'batch_size': batch_size,
            'out_channels': 8,
            'hidden_size': 512,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,
            'dropout': 0.2,
            'punishment': 5
        }
    },
    'TGCN': {
        'class': TGCN,
        'default_args': {
            'struc': 'graph',
            'seed': seed,
            'lr': 0.001,
            'epoch': 100,
            'timestep_hidden': timestep_hidden,
            'batch_size': batch_size,
            'out_channels': 8,
            'hidden_size': 512,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,            
            'dropout': 0.2,
            'punishment': 5
        }
    },
}