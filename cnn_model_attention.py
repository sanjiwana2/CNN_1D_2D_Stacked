import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN1DATT(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNN1DATT, self).__init__()
        
        # 1st convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        
        # 2nd convolutional layer
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        # 3rd convolutional layer
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        # Dynamically calculate the flattened size after conv layers
        self.flattened_size = self._get_flattened_size(input_size)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, input_size):
        """Calculate the flattened size of the conv layers' output."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            flattened_size = x.view(x.size(0), -1).size(1)
        return flattened_size
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Attention mechanism
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

class CNNLSTMATT(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNNLSTMATT, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)
        
        # Attention mechanism
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

class CNN2DATT(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNN2DATT, self).__init__()
        
        # CNN layer
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Get the CNN output size
        self.cnn_out_size = self._get_cnn_output_size(input_size)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.cnn_out_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(self.cnn_out_size, num_classes)

    def _get_cnn_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        x = self.fc(x)
        return x

class CNNLSTM2DATT(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNNLSTM2DATT, self).__init__()
        
        # CNN layer
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Get CNN output size
        self.cnn_out_size = self._get_cnn_output_size(input_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.cnn_out_size, hidden_size=32, num_layers=1, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(32, num_classes)

    def _get_cnn_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.cnn(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Apply attention on LSTM output
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        x = self.fc(x)
        return x