import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class TubularDataset(Dataset):
    def __init__(self, dataframe, feature_space, label_col_name):
        self.dataframe = dataframe
        self.features = torch.tensor(dataframe.loc[:, feature_space].values, dtype=torch.float32)
        self.labels = torch.tensor(dataframe.loc[:, label_col_name].values, dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

def cross_validation_dataloader(data, n_fold, batch_size, feature_space, label_col_name):
    kf = KFold(n_splits = n_fold, shuffle = True, random_state = 42)
    splits = []

    # Perform 5-fold cross-validation
    for train_index, test_index in kf.split(data):
        # Splitting data into training and testing
        train_val_data = data.iloc[train_index, :]
        data_test = data.iloc[test_index, :]

        # Further split the train_val_data into train and validation sets
        data_train, data_val = train_test_split(
            train_val_data,
            test_size=0.25,  # 0.25 * 0.8 = 0.2 of the total data
            random_state=42
        )

        dataset_train = TubularDataset(data_train, feature_space, label_col_name)
        dataset_val = TubularDataset(data_val, feature_space, label_col_name)
        dataset_test = TubularDataset(data_test, feature_space, label_col_name)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


        # Append the splits as tuples (train, val, test) to the list
        splits.append((train_loader, val_loader, test_loader))

    return splits