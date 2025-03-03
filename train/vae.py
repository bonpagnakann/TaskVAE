import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),

        )
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384),
            Reshape(-1, 16, 24),
            nn.Upsample(size=126),
            nn.ConvTranspose1d(16, 16, kernel_size=5, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 6, kernel_size=3, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=1)
        )


    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var

    def encode(self, x):
        x = self.encoder(x)
        mean = self.z_mean(x)
        var = self.z_log_var(x)
        return mean, var

    def decode(self, z):
        out = self.decoder(z)
        return out

    def classify(self, z):
        out = self.classifier(z)
        return out

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_log_var.size(0), z_log_var.size(1)).to(z_mu.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        encoded, z_mean, z_log_var = self.encoding_fn(x)
        decoded = self.decoder(encoded)
        class_probs = self.classifier(encoded)
        return decoded, z_mean, z_log_var, class_probs

class Cnn1(nn.Module):
    def __init__(self, output_dim):
        """
        """
        super(Cnn1, self).__init__()
        self.n_chan = 6
        self.n_classes = output_dim

        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=5, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc0 = nn.Linear(1008,100)
        self.fc = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = x.reshape(batch_size, -1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc(x)

        return x
