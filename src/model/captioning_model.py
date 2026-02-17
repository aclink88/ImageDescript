import torch
import torch.nn as nn
from src.model.encoder import EncoderCNN
from src.model.decoder import DecoderRNN

class CaptioningModel(nn.Module):
    """
    An end-to-end model that combines the CNN Encoder and Transformer Decoder.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, nhead=8, train_cnn=False):
        """
        Initializes the Encoder and Decoder sub-modules.
        """
        super(CaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_cnn)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, nhead)

    def forward(self, images, captions):
        """
        Standard forward pass for training.
        Feeds an image and its corresponding caption through the model to get predictions.
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=50):
        """
        Generates a caption for a single image during inference.
        """
        self.eval() # Set model to evaluation mode
        
        result_caption = []
        
        with torch.no_grad():
            # Get image features from the encoder
            features = self.encoder(image.unsqueeze(0))
            
            # Start the caption with the <SOS> token
            # The decoder expects a batch, so we create a batch of 1
            states = None
            inputs = torch.tensor([vocab.stoi["<SOS>"]]).unsqueeze(0).to(image.device)
            
            for _ in range(max_length):
                # The decoder's forward pass is designed for training with full sequences.
                # For inference, we need to adapt it. A proper implementation would handle
                # the evolving state, but for a simple greedy search, we can feed the
                # sequence generated so far.
                
                # A simple way for inference is to just get the last word's prediction
                # but a transformer decoder needs the whole sequence so far.
                
                # Let's get the decoder's output for the current sequence
                predictions = self.decoder(features, inputs)
                
                # We only care about the prediction for the *last* word in the sequence
                predicted_token_index = predictions[0, -1, :].argmax(dim=-1)
                
                # Append the new prediction to our inputs for the next time step
                inputs = torch.cat([inputs, predicted_token_index.unsqueeze(0).unsqueeze(0)], dim=1)

                # Stop if we predict the <EOS> token
                if predicted_token_index.item() == vocab.stoi["<EOS>"]:
                    break
                
                # Convert the token index back to a word and add it to our result
                word = vocab.itos[predicted_token_index.item()]
                result_caption.append(word)
                
        self.train() # Set model back to training mode
        return " ".join(result_caption)

if __name__ == '__main__':
    # This file is not meant to be run directly as a script.
    # It provides the final model class to be used by the training script.
    print("CaptioningModel class is defined in this file.")
    print("It should be imported and used within a training script.")

    # You could add more detailed tests here if desired, but it would require
    # creating a dummy vocabulary and dataset, which is complex for a simple test.
    pass
