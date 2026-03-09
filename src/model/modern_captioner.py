import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

class ModernCaptioner(nn.Module):
    """
    Modern Image Captioner with flexible LoRA configurations.
    Can be configured to match older trained versions or the new Dual-LoRA version.
    """
    def __init__(self, vocab_size, rank=16, use_encoder_lora=False):
        super(ModernCaptioner, self).__init__()
        
        # 1. ENCODER: Vision Transformer (ViT)
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 2. DECODER: GPT-2
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

        # 3. THE BRIDGE
        self.bridge = nn.Linear(768, 768)

        # 4. LoRA CONFIGURATION
        
        # Apply LoRA to Encoder ONLY if requested (False for your first model)
        if use_encoder_lora:
            encoder_lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=["query", "value", "key"], 
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, encoder_lora_config)
            print(f"LoRA applied to Encoder with rank {rank}.")
        
        # Apply LoRA to Decoder (Always applied in our modern models)
        decoder_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["c_attn"], 
            lora_dropout=0.1,
            bias="none",
        )
        self.decoder = get_peft_model(self.decoder, decoder_lora_config)
        print(f"LoRA applied to Decoder with rank {rank}.")

    def forward(self, images, input_ids):
        # 1. Encode image
        encoder_outputs = self.encoder(pixel_values=images)
        image_features = self.bridge(encoder_outputs.last_hidden_state)

        # 2. Get text embeddings
        inputs_embeds = self.decoder.base_model.transformer.wte(input_ids)
        
        # 3. Concatenate
        full_embeddings = torch.cat((image_features, inputs_embeds), dim=1)
        
        # 4. Decode
        outputs = self.decoder(inputs_embeds=full_embeddings)
        
        return outputs.logits

if __name__ == "__main__":
    # Test initialization
    model = ModernCaptioner(vocab_size=50257)
    print("Modern Multi-modal Model Initialized Successfully.")
