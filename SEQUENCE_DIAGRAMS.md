# Sequence Diagrams for Open Cross-Layer Transcoder

This document contains sequence diagrams illustrating the flow of operations in the Open Cross-Layer Transcoder implementation.

## 1. Initialization Flow

```mermaid
sequenceDiagram
    participant User
    participant OpenCrossLayerTranscoder
    participant GPT2Model
    participant GPT2Tokenizer
    participant Encoders
    participant Decoders
    participant ActivationFunctions
    participant Hooks

    User->>OpenCrossLayerTranscoder: __init__(model_name, num_features, device, activation_type)
    OpenCrossLayerTranscoder->>GPT2Model: from_pretrained(model_name)
    GPT2Model-->>OpenCrossLayerTranscoder: base_model
    OpenCrossLayerTranscoder->>GPT2Tokenizer: from_pretrained(model_name)
    GPT2Tokenizer-->>OpenCrossLayerTranscoder: tokenizer
    
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Freeze base_model parameters
    
    loop For each layer (0 to num_layers-1)
        OpenCrossLayerTranscoder->>Encoders: Create Linear(hidden_size, num_features)
        OpenCrossLayerTranscoder->>Decoders: Create decoders for all subsequent layers
    end
    
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: _initialize_weights()
    
    alt activation_type == "jumprelu"
        OpenCrossLayerTranscoder->>ActivationFunctions: Create JumpReLU for each layer
    else activation_type == "topk"
        OpenCrossLayerTranscoder->>ActivationFunctions: Create TopK for each layer
    else default
        OpenCrossLayerTranscoder->>ActivationFunctions: Create ReLU for each layer
    end
    
    OpenCrossLayerTranscoder->>Hooks: _register_hooks()
    loop For each layer
        Hooks->>GPT2Model: Register forward hook on MLP layer
    end
    
    OpenCrossLayerTranscoder-->>User: Initialized transcoder
```

## 2. Forward Pass (Inference) Flow

```mermaid
sequenceDiagram
    participant User
    participant OpenCrossLayerTranscoder
    participant GPT2Model
    participant MLPHooks
    participant Encoders
    participant ActivationFunctions
    participant Decoders

    User->>OpenCrossLayerTranscoder: forward(input_ids, attention_mask)
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Clear mlp_inputs_captured and mlp_activations
    
    OpenCrossLayerTranscoder->>GPT2Model: forward(input_ids, attention_mask)
    
    loop For each transformer layer
        GPT2Model->>MLPHooks: MLP forward hook triggered
        MLPHooks->>OpenCrossLayerTranscoder: Store mlp_input and mlp_output
    end
    
    GPT2Model-->>OpenCrossLayerTranscoder: outputs (last_hidden_state)
    
    loop For each layer (0 to num_layers-1)
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Get mlp_input and mlp_output from hooks
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Normalize mlp_input and mlp_output
        
        OpenCrossLayerTranscoder->>Encoders: encoder[layer_idx](mlp_input_normalized)
        Encoders-->>OpenCrossLayerTranscoder: features
        
        OpenCrossLayerTranscoder->>ActivationFunctions: activation[layer_idx](features)
        ActivationFunctions-->>OpenCrossLayerTranscoder: features_activated
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Store in feature_activations[layer_idx]
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Initialize reconstructed = zeros
        
        loop For each encoder_layer (0 to layer_idx)
            OpenCrossLayerTranscoder->>Decoders: decoder[encoder_layer][layer_idx](feature_activations[encoder_layer])
            Decoders-->>OpenCrossLayerTranscoder: contribution
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: reconstructed += contribution
        end
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Store in reconstructed_activations[layer_idx]
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Update feature_importance
    end
    
    OpenCrossLayerTranscoder-->>User: {last_hidden_state, feature_activations, reconstructed_activations}
```

## 3. Training Flow

```mermaid
sequenceDiagram
    participant User
    participant OpenCrossLayerTranscoder
    participant Tokenizer
    participant Optimizer
    participant Scheduler
    participant GPT2Model
    participant Encoders
    participant ActivationFunctions
    participant Decoders
    participant LossFunction

    User->>OpenCrossLayerTranscoder: train_transcoder(texts, batch_size, num_epochs, ...)
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Set model to training mode
    
    OpenCrossLayerTranscoder->>Optimizer: Create Adam optimizer
    OpenCrossLayerTranscoder->>Scheduler: Create ReduceLROnPlateau scheduler
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Initialize TrainingMetric
    
    OpenCrossLayerTranscoder->>Tokenizer: Tokenize all texts
    Tokenizer-->>OpenCrossLayerTranscoder: encoded_texts
    
    loop For each epoch
        loop For each batch
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Get batch of texts
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Pad texts to same length
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Create attention masks
            
            OpenCrossLayerTranscoder->>Optimizer: zero_grad()
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Clear mlp_inputs_captured and mlp_activations
            
            OpenCrossLayerTranscoder->>GPT2Model: forward(input_ids, attention_mask)
            
            Note over GPT2Model: Hooks capture MLP inputs/outputs
            
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Initialize all_features = {}
            
            loop For each layer
                OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Get mlp_input and mlp_output
                OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Normalize inputs/outputs
                
                OpenCrossLayerTranscoder->>Encoders: encoder[layer_idx](mlp_input_normalized)
                Encoders-->>OpenCrossLayerTranscoder: features
                
                OpenCrossLayerTranscoder->>ActivationFunctions: activation[layer_idx](features)
                ActivationFunctions-->>OpenCrossLayerTranscoder: features_activated
                
                OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Store in all_features[layer_idx]
                
                OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Initialize reconstructed = zeros
                
                loop For each encoder_layer (0 to layer_idx)
                    OpenCrossLayerTranscoder->>Decoders: decoder[encoder_layer][layer_idx](all_features[encoder_layer])
                    Decoders-->>OpenCrossLayerTranscoder: contribution
                    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: reconstructed += contribution
                end
                
                OpenCrossLayerTranscoder->>LossFunction: Calculate L0 metric (sparsity)
                LossFunction-->>OpenCrossLayerTranscoder: l0_metric
                
                OpenCrossLayerTranscoder->>LossFunction: MSE(reconstructed, mlp_output_normalized)
                LossFunction-->>OpenCrossLayerTranscoder: recon_loss
                
                alt activation_type != "topk"
                    OpenCrossLayerTranscoder->>LossFunction: Calculate L1 sparsity loss
                    LossFunction-->>OpenCrossLayerTranscoder: l1_loss
                else activation_type == "topk"
                    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: l1_loss = 0
                end
                
                OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: total_loss += recon_loss + l1_loss
            end
            
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: total_loss.backward()
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Clip gradients
            OpenCrossLayerTranscoder->>Optimizer: step()
            
            OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Update epoch metrics
        end
        
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Calculate average metrics
        OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Append to TrainingMetric
        OpenCrossLayerTranscoder->>Scheduler: step(avg_total_loss)
        OpenCrossLayerTranscoder->>User: Print epoch metrics
    end
    
    OpenCrossLayerTranscoder-->>User: TrainingMetric
```

## 4. ReplacementModel Forward Pass Flow

```mermaid
sequenceDiagram
    participant User
    participant ReplacementModel
    participant GPT2LMHeadModel
    participant MLPHooks
    participant OpenCrossLayerTranscoder
    participant Encoders
    participant ActivationFunctions
    participant Decoders

    User->>ReplacementModel: forward(input_ids, attention_mask)
    ReplacementModel->>ReplacementModel: Clear stored_features
    
    ReplacementModel->>GPT2LMHeadModel: forward(input_ids, attention_mask)
    
    loop For each transformer layer
        GPT2LMHeadModel->>MLPHooks: MLP forward hook triggered
        
        alt transcoder exists and layer_idx < num_layers
            MLPHooks->>ReplacementModel: Get current_mlp_input (input_args[0])
            ReplacementModel->>ReplacementModel: Normalize current_mlp_input
            
            ReplacementModel->>OpenCrossLayerTranscoder: Set to eval mode
            
            ReplacementModel->>Encoders: transcoder.encoder[layer_idx](current_mlp_input_rk)
            Encoders-->>ReplacementModel: features
            
            ReplacementModel->>ActivationFunctions: transcoder.activation[layer_idx](features)
            ActivationFunctions-->>ReplacementModel: features_activated
            
            ReplacementModel->>ReplacementModel: Store in stored_features[layer_idx]
            
            ReplacementModel->>ReplacementModel: Initialize reconstructed = zeros
            
            loop For each encoder_layer (0 to layer_idx)
                ReplacementModel->>Decoders: transcoder.decoder[encoder_layer][layer_idx](stored_features[encoder_layer])
                Decoders-->>ReplacementModel: contribution
                ReplacementModel->>ReplacementModel: reconstructed += contribution
            end
            
            MLPHooks-->>GPT2LMHeadModel: Return reconstructed (instead of original MLP output)
        else no transcoder or invalid layer
            MLPHooks-->>GPT2LMHeadModel: Return original output_tensor
        end
    end
    
    GPT2LMHeadModel-->>ReplacementModel: outputs
    ReplacementModel-->>User: outputs
```

## 5. Text Generation with ReplacementModel

```mermaid
sequenceDiagram
    participant User
    participant ReplacementModel
    participant Tokenizer
    participant GPT2LMHeadModel
    participant MLPHooks
    participant OpenCrossLayerTranscoder

    User->>ReplacementModel: generate(text, max_length)
    ReplacementModel->>Tokenizer: encode(text, return_tensors="pt")
    Tokenizer-->>ReplacementModel: input_ids
    
    ReplacementModel->>GPT2LMHeadModel: generate(input_ids, max_length, ...)
    
    Note over GPT2LMHeadModel: During generation, hooks intercept MLP outputs
    
    loop For each generation step
        GPT2LMHeadModel->>MLPHooks: MLP forward hook triggered
        MLPHooks->>OpenCrossLayerTranscoder: Use transcoder to reconstruct MLP output
        OpenCrossLayerTranscoder-->>MLPHooks: reconstructed output
        MLPHooks-->>GPT2LMHeadModel: Use reconstructed output
    end
    
    GPT2LMHeadModel-->>ReplacementModel: output_ids
    ReplacementModel->>Tokenizer: decode(output_ids[0], skip_special_tokens=True)
    Tokenizer-->>ReplacementModel: output_text
    ReplacementModel-->>User: Generated text
```

## 6. Feature Activation Extraction Flow

```mermaid
sequenceDiagram
    participant User
    participant OpenCrossLayerTranscoder
    participant Tokenizer
    participant ForwardMethod

    User->>OpenCrossLayerTranscoder: get_feature_activations(text)
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Set to eval mode
    OpenCrossLayerTranscoder->>Tokenizer: encode(text, return_tensors="pt")
    Tokenizer-->>OpenCrossLayerTranscoder: input_ids
    
    OpenCrossLayerTranscoder->>ForwardMethod: forward(input_ids)
    Note over ForwardMethod: Executes forward pass flow (see diagram 2)
    ForwardMethod-->>OpenCrossLayerTranscoder: outputs
    
    OpenCrossLayerTranscoder->>OpenCrossLayerTranscoder: Extract feature_activations from outputs
    OpenCrossLayerTranscoder-->>User: feature_activations (Dict[int, Tensor])
```

## Key Components

### Hooks System
- **Purpose**: Capture MLP inputs and outputs during forward pass
- **Registration**: Done in `_register_hooks()` during initialization
- **Storage**: MLP inputs/outputs stored in `mlp_inputs_captured` and `mlp_activations` dictionaries

### Cross-Layer Architecture
- **Encoders**: Map MLP inputs to interpretable features (one per layer)
- **Decoders**: Map features back to MLP output space (one per encoder-decoder layer pair)
- **Activation Functions**: Apply sparsity (JumpReLU, TopK, or ReLU)

### Training Process
- **Loss Components**:
  - Reconstruction Loss: MSE between reconstructed and actual MLP outputs
  - Sparsity Loss: L1 regularization on features (for non-TopK activations)
- **Optimization**: Adam optimizer with learning rate scheduling

### Replacement Model
- **Purpose**: Replace MLP outputs with transcoder reconstructions during inference
- **Mechanism**: Uses forward hooks to intercept and replace MLP outputs in real-time

