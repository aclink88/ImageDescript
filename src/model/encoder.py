import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN-based Encoder to extract features from images.
    We use a pre-trained ResNet model and replace the final fully connected
    layer with a new linear layer to output embeddings of a desired size.
    """
    def __init__(self, embed_size, train_cnn=False):
        """
        Args:
            embed_size (int): The size of the embedding vector.
            train_cnn (bool): Whether to fine-tune the pre-trained CNN layers.
        """
        super(EncoderCNN, self).__init__()
        
        # Load a pre-trained ResNet model (e.g., resnet50)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all layers of the CNN if we are not fine-tuning
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn
            
        # The original ResNet's fully connected layer ('fc') has an output size
        # of 1000 (for ImageNet classes). We need to replace it.
        # Get the number of input features to the fc layer.
        in_features = self.resnet.fc.in_features
        
        # Replace the fully connected layer with a new linear layer that
        # outputs vectors of our desired embed_size.
        self.resnet.fc = nn.Linear(in_features, embed_size)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, images):
        """
        Forward pass to extract features from a batch of images.
        """
        # Pass images through the ResNet model (excluding the final original fc layer)
        features = self.resnet(images)
        
        # Apply ReLU and Dropout
        return self.dropout(self.relu(features))

if __name__ == '__main__':
    # A simple test to verify the encoder's output shape
    embed_size = 256
    encoder = EncoderCNN(embed_size)
    
    # Create a dummy image batch (e.g., batch_size=5, 3 channels, 224x224)
    dummy_images = torch.randn(5, 3, 224, 224)
    
    # Get the features
    features = encoder(dummy_images)
    
    print(f"Input images shape: {dummy_images.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Expected output shape: (5, {embed_size})")
    
    assert features.shape == (5, embed_size)
    print("\nEncoder test passed!")
