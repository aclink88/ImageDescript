import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

class ModernCaptioner(nn.Module):
    """
    Advanced Modern Image Captioner using Dual-LoRA fine-tuning.
    Both the Vision Transformer (ViT) and GPT-2 are optimized using LoRA.
    """
    def __init__(self, vocab_size, rank=64):
        super(ModernCaptioner, self).__init__()
        
        # 1. ENCODER: Vision Transformer (ViT)
        self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 2. DECODER: GPT-2
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.decoder.resize_token_embeddings(vocab_size)

        # 3. THE BRIDGE (Connecting Vision to Language)
        self.bridge = nn.Linear(768, 768)

        # 4. LoRA CONFIGURATION
        # rank=64 allows for more complex learning capacity.
        encoder_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["query", "value", "key"], 
            lora_dropout=0.1,
            bias="none",
        )
        
        decoder_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["c_attn"], 
            lora_dropout=0.1,
            bias="none",
        )
        
        # Apply LoRA to both components
        self.encoder = get_peft_model(self.encoder, encoder_lora_config)
        self.decoder = get_peft_model(self.decoder, decoder_lora_config)
        
        print(f"Dual-LoRA initialized with rank {rank}.")

    def forward(self, images, input_ids):
        """
        Standard forward pass for training.
        """
        # STEP 1: Encode image into 197 visual tokens
        encoder_outputs = self.encoder(pixel_values=images)
        image_features = self.bridge(encoder_outputs.last_hidden_state) 

        # STEP 2: Get word embeddings for the text
        inputs_embeds = self.decoder.base_model.transformer.wte(input_ids) 
        
        # STEP 3: Concatenate (Visual Prompting)
        full_embeddings = torch.cat((image_features, inputs_embeds), dim=1) 
        
        # STEP 4: Decode with GPT-2
        outputs = self.decoder(inputs_embeds=full_embeddings)
        
        return outputs.logits

    def generate_caption_beam(self, image, tokenizer, beam_width=5, max_length=60, min_length=5, repetition_penalty=1.2, length_penalty=1.0):
        """
        Generates a caption using Beam Search with improved robustness against empty outputs.
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 1. Encode Image
            encoder_outputs = self.encoder(pixel_values=image.unsqueeze(0))
            image_features = self.bridge(encoder_outputs.last_hidden_state)

            # 2. Initialize Beams
            # Use the standard GPT-2 end-of-text token as the starting point
            start_token = tokenizer.eos_token_id
            beams = [([start_token], 0.0)]
            
            for i in range(max_length):
                candidates = []
                
                for seq, score in beams:
                    # If already at EOS, keep it as is
                    if seq[-1] == tokenizer.eos_token_id and i > 0:
                        candidates.append((seq, score))
                        continue
                    
                    # Prepare inputs
                    input_ids = torch.tensor([seq]).to(device)
                    inputs_embeds = self.decoder.base_model.transformer.wte(input_ids)
                    full_embeddings = torch.cat((image_features, inputs_embeds), dim=1)
                    
                    logits = self.decoder(inputs_embeds=full_embeddings).logits
                    next_token_logits = logits[0, -1, :]
                    
                    # Apply repetition penalty
                    for token_id in set(seq):
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty
                    
                    # FORCE MINIMUM LENGTH: 
                    # If we haven't reached min_length, make the EOS token probability extremely low.
                    if i < min_length:
                        next_token_logits[tokenizer.eos_token_id] = -1e20

                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width)
                    
                    for j in range(beam_width):
                        candidates.append((
                            seq + [top_k_tokens[j].item()], 
                            score + top_k_log_probs[j].item()
                        ))
                
                # Sort and apply Length Penalty (score / length^penalty)
                # This prevents the model from always favoring shorter sequences
                beams = sorted(candidates, key=lambda x: x[1] / (len(x[0])**length_penalty), reverse=True)[:beam_width]
                
                if all(s[-1] == tokenizer.eos_token_id for s, _ in beams):
                    break
            
            # 3. Return best sequence
            best_seq = beams[0][0]
            return tokenizer.decode(best_seq, skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Test initialization
    model = ModernCaptioner(vocab_size=50257)
    print("Modern Multi-modal Model with Beam Search Initialized.")
