from torch.utils.data import DataLoader

from .datasets import Dataset_CLS_encoded, Dataset_IMP_encoded, Dataset_IMP_Pred_encoded

dataset_dict = {
    'classification': {
        'encoded': {
            'train': Dataset_CLS_encoded,
            'test': Dataset_CLS_encoded
        }
    },
    'imputation': {
        'encoded': {
            'pred': Dataset_IMP_Pred_encoded,
            'train': Dataset_IMP_encoded,
            'test': Dataset_IMP_encoded
        }
    }
}


def get_loader(config, 
               flag='train'):
    root_path = config.data.root_path
    task_name = config.model.task_name
    subject_list = config.data.train_subjects if flag == 'train' else config.data.test_subjects

    timeenc = 0 if config.model.embed != 'timeF' else 1
    size = [config.model.seq_len, config.model.label_len, config.model.pred_len]
    
    drop_last = True if flag == 'train' else False
    shuffle = True if flag == 'train' else False
    batch_size = 1 if flag == 'pred' else config.train.batch_size

    data_kwargs = {
        'root_path': root_path,
        'flag': flag,
        'size': size,
        'timeenc': timeenc,
        'freq': 'h'
    }
    if task_name == 'imputation' and flag == 'pred':
        data_kwargs['subject'] = subject_list[0]
        data_kwargs['act'] = config.data.test_act
    else:
        data_kwargs['subjects'] = subject_list
        
    dataset = dataset_dict[task_name][config.data.type][flag](**data_kwargs)
    
    print(f'Loaded: {len(dataset)} {flag} samples.')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last
    )
    
    return dataloader
    

    