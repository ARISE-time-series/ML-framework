from torch.utils.data import DataLoader, Subset

from .datasets import Dataset_CLS_encoded, Dataset_IMP_encoded, Dataset_IMP_Pred_encoded

dataset_dict = {
    'classification': {
        'encoded': {
            'train': Dataset_CLS_encoded,
            'test': Dataset_CLS_encoded, 
            'explain': Dataset_CLS_encoded
        }
    },
    'imputation': {
        'encoded': {
            'pred': Dataset_IMP_Pred_encoded,
            'train': Dataset_IMP_encoded,
            'test': Dataset_IMP_encoded
        }
    },
    'forecast': {
        'encoded': {
            'pred': Dataset_IMP_Pred_encoded,
            'train': Dataset_IMP_encoded,
            'test': Dataset_IMP_encoded
        }, 
        'manual': {
            
            
        } 
    }
}


def get_loader(config, 
               flag='train', 
               subject=None,
               act=None):
    root_path = config.data.root_path
    task_name = config.model.task_name

    subject_list = config.data.train_subjects if flag == 'train' else config.data.test_subjects

    timeenc = 1 if config.model.embed == 'timeF' else 0
    size = [config.model.seq_len, config.model.label_len, config.model.pred_len]
    
    drop_last = True if flag == 'train' else False
    shuffle = True if flag == 'train' or flag == 'explain' else False
    batch_size = 1 if flag == 'pred' else config.train.batch_size
    freq = config.model.freq
    # batch_size = config.train.batch_size
    data_kwargs = {
        'root_path': root_path,
        'encode_dir': config.data.encode_dir,
        'flag': flag,
        'cols': config.data.cols,
        'size': size,
        'timeenc': timeenc,
        'freq': freq, 
        'scale': config.data.scale,
        'embedding': config.data.embedding, 
    }
    if (task_name == 'imputation' or task_name == 'forecast') and flag == 'pred':
        data_kwargs['subject'] = subject
        data_kwargs['act'] = act
    else:
        data_kwargs['subjects'] = subject_list
        data_kwargs['acts'] = config.data.train_acts
        
    dataset = dataset_dict[task_name][config.data.type][flag](**data_kwargs)
    subset_ratio = config.data.subset if flag == 'train' else 1
    subset = Subset(dataset, list(range(0, int(len(dataset) * subset_ratio))))
    print(f'Loaded: {len(subset)} {flag} samples.')

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last
    )
    
    return dataloader
    

    