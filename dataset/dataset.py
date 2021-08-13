import yaml
from gandlf_data import GANDLFData

if __name__ == '__main__':
    config = yaml.load(open('conf.yml'), Loader=yaml.FullLoader)
    g = GANDLFData(data_path="data", **config)
    train_loader = g.train_loader
    for batch in train_loader:
        print(batch.keys())  # dict_keys(['subject_id', '1', '2', '3', '4', 'label', 'path_to_metadata', 'index_ini'])
        print(batch['1']['data'].shape)  # 1, 128, 128, 128

        # Note : need to go from 4 x 1, 128, 128, 128 to 4, 128, 128, 128

        print(batch['label']['data'].shape)  # 1, 128, 128, 128
