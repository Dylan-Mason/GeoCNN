import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm  

image_dir = 'Newdataset'
data_file = 'test.csv'

# We load up the data and preprocess
data_df = pd.read_csv(data_file)
categorical_columns = ['region', 'sub-region', 'drive_side', 'climate', 'soil', 'land_cover']
data_df[categorical_columns] = data_df[categorical_columns].apply(lambda x: x.astype('category').cat.codes).fillna(-1)

# Set up dictionaries
country_to_index, city_to_index = {}, {}
index_to_country, index_to_city = {}, {}

def get_or_add_mapping(mapping, reverse_mapping, name):
    if name not in mapping:
        idx = len(mapping)
        mapping[name], reverse_mapping[idx] = idx, name
    return mapping[name]

# Map the countries and cities to make sure they match
data_df['country_index'] = data_df['country'].apply(lambda x: get_or_add_mapping(country_to_index, index_to_country, x))
data_df['city_index'] = data_df['city'].apply(lambda x: get_or_add_mapping(city_to_index, index_to_city, x))

class ImageCountryDataset(Dataset):
    def __init__(self, image_dir, data_df, img_size, start_idx=10001, transform=None):
        self.image_dir = image_dir
        self.data_df = data_df.iloc[start_idx:int(start_idx + 0.2 * len(data_df))]  # Limit to 20% of dataset
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        actual_idx = idx + 10001
        country_index = self.data_df.iloc[idx]['country_index']
        city_index = self.data_df.iloc[idx]['city_index']
        additional_features = torch.tensor(self.data_df.iloc[idx][categorical_columns].astype(float).values, dtype=torch.float32)
        
        img_path = os.path.join(self.image_dir, f"{actual_idx}.png")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Missing image at {img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        if self.transform:
            image = self.transform(image)

        return image, additional_features, (country_index, city_index)
    
resolutions = [256]

# Define transformation function for flipping, color changing, etc
def get_transform(resolution):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def objective(trial, architecture, resolution, data_df, train_loader, val_loader, num_countries, num_cities):
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    class MultiInputModel(nn.Module):
        def __init__(self, architecture, num_countries, num_cities, additional_features_dim):
            super().__init__()
            weights = models.ResNet50_Weights.IMAGENET1K_V1

            self.resnet = architecture(weights=weights)
            
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
            
            self.additional_fc = nn.Sequential(
                nn.Linear(additional_features_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64)
            )
            
            self.final_fc = nn.Linear(in_features + 64, num_countries + num_cities)

        def forward(self, image, additional_features):
            img_features = self.resnet(image)
            additional_features = self.additional_fc(additional_features)
            combined_features = torch.cat((img_features, additional_features), dim=1)
            return self.final_fc(combined_features)
    # checks if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiInputModel(architecture, num_countries, num_cities, len(categorical_columns)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    country_loss_fn = city_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 5
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        # Starts the training with the progress bar since this thing runs for hours
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            if batch is None:  # Skips the empty batches
                continue
            images, additional_features, (country_labels, city_labels) = batch
            images, additional_features = images.to(device).float(), additional_features.to(device)
            country_labels, city_labels = country_labels.to(device).long(), city_labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images, additional_features)
            country_loss = country_loss_fn(outputs[:, :num_countries], country_labels)
            city_loss = city_loss_fn(outputs[:, num_countries:], city_labels)
            loss = country_loss + city_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            return float('inf')

        avg_train_loss = running_loss / batch_count

        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        # Starts the validation loop
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                if batch is None:
                    continue
                images, additional_features, (country_labels, city_labels) = batch
                images, additional_features = images.to(device).float(), additional_features.to(device)
                country_labels, city_labels = country_labels.to(device).long(), city_labels.to(device).long()
                outputs = model(images, additional_features)
                val_loss += (country_loss_fn(outputs[:, :num_countries], country_labels) + 
                             city_loss_fn(outputs[:, num_countries:], city_labels)).item()
                val_batch_count += 1

        if val_batch_count == 0:
            return float('inf')

        avg_val_loss = val_loss / val_batch_count
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss

results = {}

# Loops through each resolution for ResNet50
model_name, model_fn = 'resnet50', models.resnet50
for resolution in resolutions:
    print(f"Starting trials for {model_name} at resolution {resolution}...")

    transform = get_transform(resolution)
    
    # Preps the dataset and dataLoader
    dataset = ImageCountryDataset(image_dir, data_df, resolution, transform=transform)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Run the Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_fn, resolution, data_df, train_loader, val_loader, 
                                           len(country_to_index), len(city_to_index)), n_trials=30)
    
    # Stores the best trial for the epoch run
    results[f"{model_name}_{resolution}"] = {
        "best_params": study.best_trial.params,
        "best_val_loss": study.best_trial.value
    }

    # Print checkpoint after each combination is tested
    print(f"Completed trials for {model_name} at resolution {resolution}.")
    print(f"Best Parameters: {study.best_trial.params}")
    print(f"Best Validation Loss: {study.best_trial.value}")
    print("="*30)

print("Final Results for ResNet50 Model with Different Resolutions")
for key, result in results.items():
    print(f"Resolution: {key}")
    print("Best Parameters:", result["best_params"])
    print("Best Validation Loss:", result["best_val_loss"])
    print("="*30)
