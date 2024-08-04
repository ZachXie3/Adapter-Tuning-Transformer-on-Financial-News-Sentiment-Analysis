from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd


# Define a custom dataset
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels = data['sentiment'].values
        self.texts = data['text'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


def load_data(batch_size=64, random_state=123456):

    # Create an instance of the dataset
    csv_file_path = 'data/all-data.csv'
    data = pd.read_csv(csv_file_path, encoding="unicode_escape", names=['sentiment', 'text'])
    # Mapping from label strings to numeric values
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    data['sentiment'] = data['sentiment'].map(label_map)

    # 70% for training, 15% for validation, 15% for test.
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)

    train_dataset = SentimentDataset(train_data)
    val_dataset = SentimentDataset(val_data)
    test_dataset = SentimentDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
