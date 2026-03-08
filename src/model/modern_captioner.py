import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel, ViTConfig, GPT2Config
from peft import LoraConfig, get_peft_model

class ModernCaptioner(nn.Module):
    """
    A State-of-the-Art Image Captioner using a Vision Transformer (ViT) encoder
    and a GPT-2 decoder, enhanced with LoRA for efficient fine-tuning.
    """
    def __init__(self, vocab_size):
        super(ModernCaptioner, self).__init__()
        
        # 1. ENCODER: Vision Transformer (ViT)
        # SOTA REMINDER: From scratch, we'd implement the 'PatchEmbedding' here.
        # This model takes a 224x224 image and turns it into 197 tokens (16x16 patches + 1 CLS token).
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 2. DECODER: GPT-2
        # GPT-2 is a decoder-only transformer. We will add cross-attention to it 
        # so it can 'look' at the ViT's output patches.
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

        # 3. THE BRIDGE (Projection Layer)
        # ViT output is 768-dim, GPT-2 input is 768-dim. If they were different,
        # we would need a linear layer here to align their dimensions.
        self.bridge = nn.Linear(768, 768)

        # 4. LoRA CONFIGURATION (The 'MLE' optimization)
        # MATH: Instead of training 100M+ parameters, we add tiny 'A' and 'B' matrices
        # to the attention layers. r=8 is the 'rank' (the width of the tiny matrices).
        config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["c_attn", "query", "value"], # Target the attention layers
            lora_dropout=0.05,
            bias="none",
        )
        
        # Apply LoRA to the decoder (the part we are fine-tuning for language)
        self.decoder = get_peft_model(self.decoder, config)
        print(f"LoRA applied. Trainable parameters reduced significantly.")

    def forward(self, images, input_ids, attention_mask=None):
        """
        Forward pass for the multi-modal model.
        """
        # STEP 1: Encode the image into patches
        # output.last_hidden_state shape: (batch, 197, 768)
        encoder_outputs = self.encoder(pixel_values=images)
        image_features = self.bridge(encoder_outputs.last_hidden_state)

        # STEP 2: Decode the text using image features as context
        # In a more complex model (like BLIP or LLaVA), we would use specific 
        # cross-attention layers. For this 'Modern Lite' version, we concatenate 
        # the image features to the front of the text embeddings.
        
        # This is a 'Prompt Tuning' style approach: The model sees 197 'visual words'
        # followed by your actual text tokens.
        
        # Get text embeddings from GPT-2
        inputs_embeds = self.decoder.transformer.wte(input_ids) # (batch, seq_len, 768)
        
        # Concatenate: [Image Patches] + [Text Tokens]
        # full_embeddings shape: (batch, 197 + seq_len, 768)
        full_embeddings = torch.cat((image_features, inputs_embeds), dim=1)
        
        # Pass everything to the GPT-2 decoder
        outputs = self.decoder(inputs_embeds=full_embeddings, attention_mask=None)
        
        return outputs.logits

if __name__ == "__main__":
    # Test initialization
    model = ModernCaptioner(vocab_size=5000)
    print("Modern Multi-modal Model Initialized Successfully.")
