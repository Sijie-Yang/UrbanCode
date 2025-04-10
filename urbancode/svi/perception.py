import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os
from tqdm.auto import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def comfort(img_path, model_path, initial_features=None, device=None):
    """
    Predict comfort score for an image using a pre-trained model.

    Args:
        img_path (str): Path to the image file
        model_path (str): Path to the pre-trained model
        initial_features (numpy.ndarray, optional): Initial feature vector
        device (str, optional): Device to run the model on ('cuda' or 'cpu')

    Returns:
        float: Predicted comfort score

    Raises:
        FileNotFoundError: If the image file or model file does not exist
        ValueError: If the model cannot be loaded
    """
    # Check if files exist
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    try:
        model = TwoStageNNModel(num_initial_features=initial_features.shape[0] if initial_features is not None else 0, 
                               num_features=20)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

    # Load and preprocess image using a deterministic transform for prediction
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Failed to load or preprocess image: {str(e)}")

    # Prepare initial features
    if initial_features is not None:
        initial_features = torch.tensor(initial_features, dtype=torch.float).unsqueeze(0).to(device)
    else:
        initial_features = torch.zeros((1, 0), dtype=torch.float).to(device)

    # Predict
    with torch.no_grad():
        try:
            _, comfort_score = model(image, initial_features)
            comfort_score = comfort_score.squeeze().item()
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    return comfort_score

class CustomDataset(Dataset):
    def __init__(self, csv_path, img_folder, 
                 img_col_name,           # Image filename column name
                 target_col_name,        # Target column name
                 feature_cols,           # Middle feature column range (start, end)
                 initial_feature_cols,    # Initial feature column range (start, end)
                 transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.img_col_name = img_col_name
        self.target_col_name = target_col_name
        self.feature_cols = feature_cols
        self.initial_feature_cols = initial_feature_cols
        
        # Default transform if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Fix StandardScaler warning by providing feature names
        initial_features_df = self.df.iloc[:, initial_feature_cols[0]:initial_feature_cols[1]]
        self.feature_names = initial_features_df.columns.tolist()
        self.scaler = StandardScaler()
        self.scaler.fit(initial_features_df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image
        img_name = self.df.iloc[idx][self.img_col_name]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Fix StandardScaler warning by using DataFrame with column names
        initial_features = self.df.iloc[idx, 
            self.initial_feature_cols[0]:self.initial_feature_cols[1]].values.astype(float)
        initial_features_df = pd.DataFrame([initial_features], columns=self.feature_names)
        initial_features = self.scaler.transform(initial_features_df)
        initial_features = torch.tensor(initial_features, dtype=torch.float).squeeze()

        # Get middle features (labels)
        numpy_data = self.df.iloc[idx, 
            self.feature_cols[0]:self.feature_cols[1]].values.astype(float)
        labels = torch.tensor(numpy_data, dtype=torch.float)
        
        # Fix deprecated Series.__getitem__ warning
        target = torch.tensor(self.df.iloc[idx].loc[self.target_col_name], dtype=torch.float)
        
        return image, initial_features, labels, target

class TwoStageNNModel(nn.Module):
    """
    A two-stage neural network model for perception tasks.
    
    This model consists of a pre-trained ResNet50 backbone followed by task-specific layers
    and a final layer for comfort prediction.
    """
    def __init__(self, num_initial_features, num_features):
        super(TwoStageNNModel, self).__init__()
        # Fix pretrained deprecation warning
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.task_layers = nn.Sequential(
            nn.Linear(self.base_features + num_initial_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_features)
        )

        self.final_layer = nn.Sequential(
            nn.Linear(self.base_features + num_initial_features + num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, initial_features):
        base_output = self.base_model(x)
        combined_input = torch.cat((base_output, initial_features), dim=1)
        features = self.task_layers(combined_input)
        combined_features = torch.cat((base_output, initial_features, features), dim=1)
        comfort = self.final_layer(combined_features)
        return features, comfort

def compute_metrics(true, pred):
    """Compute regression metrics"""
    true = np.array(true)
    pred = np.array(pred)
    
    # Basic metrics
    mse = ((true - pred) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(true - pred).mean()
    
    # R2 score
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

class TwoStageNNPerception:
    """
    A class for training and evaluating a two-stage neural network for perception tasks.
    
    This class handles dataset creation, model training, validation, and testing.
    
    Example:
        >>> perception = TwoStageNNPerception(
        ...     data_csv_path='data.csv',
        ...     image_folder='images/',
        ...     model_save_path='models/',
        ...     img_col_name='image',
        ...     target_col_name='comfort',
        ...     feature_cols=(10, 20),
        ...     initial_feature_cols=(0, 10)
        ... )
        >>> perception.train(num_epochs=10)
    """
    def __init__(self, 
                 data_csv_path,
                 image_folder,
                 model_save_path,
                 img_col_name,           # Image filename column name
                 target_col_name,        # Target column name
                 feature_cols,           # Middle feature column range (start, end)
                 initial_feature_cols,    # Initial feature column range (start, end)
                 train_ratio=0.6,
                 val_ratio=0.2,
                 random_seed=42,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Calculate number of features
        num_initial_features = initial_feature_cols[1] - initial_feature_cols[0]
        num_features = feature_cols[1] - feature_cols[0]
        
        # Initialize dataset with updated parameter names
        dataset = CustomDataset(
            csv_path=data_csv_path, 
            img_folder=image_folder,
            img_col_name=img_col_name,
            target_col_name=target_col_name,
            feature_cols=feature_cols,
            initial_feature_cols=initial_feature_cols
        )
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Initialize model
        self.model = TwoStageNNModel(num_initial_features, num_features).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # Fix verbose deprecation warning
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'logs'))
        
    def train(self, num_epochs):
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train for
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            self.model.eval()
            val_loss = self._validate(epoch)
            
            # Test phase
            test_metrics = self._test(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(self.model_save_path, 'best_model.pth'))
                
            # Print progress with all metrics
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print("Test Metrics:")
            print(f"  MSE:  {test_metrics['mse']:.4f}")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
            print(f"  MAE:  {test_metrics['mae']:.4f}")
            print(f"  R2:   {test_metrics['r2']:.4f}")
            print("-" * 50)
            
        self.writer.close()
        
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        epoch_loss = 0
        for images, initial_features, labels, target in tqdm(self.train_loader, 
                                                           desc="Training"):
            images = images.to(self.device)
            initial_features = initial_features.to(self.device)
            labels = labels.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            features, comfort = self.model(images, initial_features)
            
            loss1 = sum(self.criterion(features[:, i], labels[:, i]) 
                       for i in range(labels.shape[1]))
            loss2 = self.criterion(comfort.squeeze(), target)
            combined_loss = torch.relu(self.model.w) * loss1 + \
                          (1 - torch.relu(self.model.w)) * loss2
            
            combined_loss.backward()
            self.optimizer.step()
            
            epoch_loss += combined_loss.item()
            
        return epoch_loss / len(self.train_loader)
    
    def _validate(self, epoch):
        """Validate the model."""
        val_loss = 0
        with torch.no_grad():
            for images, initial_features, labels, target in tqdm(self.val_loader, 
                                                               desc="Validating"):
                images = images.to(self.device)
                initial_features = initial_features.to(self.device)
                labels = labels.to(self.device)
                target = target.to(self.device)
                
                features, comfort = self.model(images, initial_features)
                
                loss1 = sum(self.criterion(features[:, i], labels[:, i]) 
                           for i in range(labels.shape[1]))
                loss2 = self.criterion(comfort.squeeze(), target)
                combined_loss = torch.relu(self.model.w) * loss1 + \
                              (1 - torch.relu(self.model.w)) * loss2
                
                val_loss += combined_loss.item()
                
        return val_loss / len(self.val_loader)
    
    def _test(self, epoch):
        """Test the model."""
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for images, initial_features, labels, target in tqdm(self.test_loader, 
                                                               desc="Testing"):
                images = images.to(self.device)
                initial_features = initial_features.to(self.device)
                
                _, comfort = self.model(images, initial_features)
                
                all_targets.extend(target.numpy())
                all_predictions.extend(comfort.squeeze().cpu().numpy())
        
        return compute_metrics(all_targets, all_predictions)
    
    def load_best_model(self):
        """Load the best model from the saved path."""
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'best_model.pth')))
        self.model.eval() 