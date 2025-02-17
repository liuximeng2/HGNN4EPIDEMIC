from model.stgcn import THGNN, TGCN
from model.dynamic_model import DTHGNN, DTGNN, FDTHGNN
from model.astgcn import ASTGCN
from model.mstgcn import MSTGCN
from model.msthgnn import MSTHGNN

batch_size = 1
seed = 0

model_dict = {
    'THGNN': {
        'class': THGNN,
        'default_args': {
            'struc': 'hgraph',
            'seed': seed,
            'lr': 0.001,
            'epoch': 100,
            'batch_size': batch_size,
            'out_channels': 8,
            'hidden_size': 256,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,
            'dropout': 0.5,
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
            'batch_size': batch_size,
            'out_channels': 8,
            'hidden_size': 256,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,            
            'dropout': 0.5,
            'punishment': 5
        }
    },
    'DTHGNN': {
        'class': DTHGNN,
        'default_args': {
            'seed': seed,
            'batch_size': batch_size,
            'in_channels': 3,
            'temporal_layer': 2,
            'spatial_layer': 1,            
            'dropout': 0.5,
            'time_strides': 1
        }
    },
    'DTGNN': {
        'class': DTGNN,
        'default_args': {
            'seed': seed,
            'lr': 0.0005,
            'batch_size': batch_size,
            'in_channels': 3,
            'out_channels': 8,
            'hidden_channels': 256,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,            
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'time_strides': 1,
            'nb_time_filter': 5
        }
    },
    'ASTGCN': {
        'class': ASTGCN,
        'default_args': {
            'seed': seed,
            'lr': 0.0005,
            'batch_size': batch_size,      
            'weight_decay': 0.0,
            'dropout': 0.5,
            'nb_block': 2,
            'in_channels': 3,
            'K': 3,
            'nb_chev_filter': 256,
            'nb_time_filter': 5,
            'time_strides': 1,
            'num_of_vertices': 2500,
        }
    },
    'MSTGCN': {
        'class': MSTGCN,
        'default_args': {
            'seed': seed,
            'batch_size': batch_size,      
            'dropout': 0.5,
            'nb_block': 2,
            'in_channels': 3,
            'K': 3,
            'nb_chev_filter': 256,
            'nb_time_filter': 5,
            'time_strides': 1,
            'num_of_vertices': 2500,
        }
    },
    'MSTHGNN': {
        'class': MSTHGNN,
        'default_args': {
            'seed': seed,
            'lr': 0.0005,
            'batch_size': batch_size,      
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'nb_block': 2,
            'in_channels': 3,
            'K': 3,
            'nb_chev_filter': 256,
            'nb_time_filter': 5,
            'time_strides': 1,
            'num_of_vertices': 2500,
        }
    },
    'FDTHGNN': {
        'class': FDTHGNN,
        'default_args': {
            'seed': seed,
            'lr': 0.0005,
            'batch_size': batch_size,
            'in_channels': 3,
            'out_channels': 8,
            'hidden_channels': 256,
            'mlp_layer': 2,
            'temporal_layer': 2,
            'spatial_layer': 1,            
            'weight_decay': 1e-4,
            'dropout': 0.5,
            'time_strides': 1,
            'nb_time_filter': 3
        }
    }
}
