import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# Predefined list of base models with local paths
BASE_MODELS = [
    ("models/gemma-3-4b-it", "Google Gemma 3 4B IT"),
    ("models/gemma-3-1b-it", "Google Gemma 3 1B IT"),
    ("models/llama-3-8b-instruct", "Meta Llama 3 8B Instruct")
]

# Helper: Map display name to model id
MODEL_ID_MAP = {v: k for k, v in BASE_MODELS}

def check_model_availability(model_path):
    """Check if a model is available locally"""
    return os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json"))

def validate_csv(file):
    if file is None:
        return gr.update(value=None), "Please upload a CSV file."
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return gr.update(value=None), f"Error reading CSV: {e}"
    cols = df.columns.str.lower()
    if ("text" in cols) or ("question" in cols and "answer" in cols):
        return gr.update(value=file), "CSV validated successfully!"
    return gr.update(value=None), "CSV must have a 'text' column or both 'question' and 'answer' columns."

def prepare_dataset(file):
    df = pd.read_csv(file.name)
    cols = df.columns.str.lower()
    if "text" in cols:
        return Dataset.from_pandas(df[["text"]].rename(columns={"text": "text"}))
    elif "question" in cols and "answer" in cols:
        # For IT models, format as proper chat messages
        formatted_texts = []
        for _, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            # Format as chat template for IT models
            chat_text = f'<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>assistant\n{answer}<end_of_turn>'
            formatted_texts.append(chat_text)
        
        return Dataset.from_pandas(pd.DataFrame({"text": formatted_texts}))
    else:
        raise ValueError("CSV must have a 'text' column or both 'question' and 'answer' columns.")

def lora_train(dataset_file, model_choice, lora_rank, epochs, learning_rate, hf_token, save_path_val, progress=gr.Progress(track_tqdm=True)):
    import os
    import time
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling
    from huggingface_hub import hf_hub_download
    log_msgs = []
    try:
        log_msgs.append("Preparing dataset...")
        ds = prepare_dataset(dataset_file)
        log_msgs.append(f"Loaded {len(ds)} samples.")
        model_id = MODEL_ID_MAP[model_choice]
        log_msgs.append(f"Loading model: {model_id}")
        token_arg = {"token": hf_token} if hf_token else {}
        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
            fp16 = False
            bf16 = True
        elif torch.cuda.is_available():
            device = "cuda"
            fp16 = True
            bf16 = False
        else:
            device = "cpu"
            fp16 = False
            bf16 = False
        log_msgs.append(f"Using device: {device}")
        log_msgs.append(f"Precision: fp16={fp16}, bf16={bf16}")
        print(f"[LoRA Lab] Using device: {device}")
        print(f"[LoRA Lab] Precision: fp16={fp16}, bf16={bf16}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **token_arg)
            # Use eager attention for Gemma models
            model_kwargs = {"torch_dtype": torch.float16 if device != "cpu" else torch.float32, **token_arg}
            if "gemma" in model_id.lower():
                model_kwargs["attn_implementation"] = "eager"
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            model.to(device)
            
            # Log available modules for debugging
            module_names = [name for name, _ in model.named_modules()]
            log_msgs.append(f"Available modules (first 20): {module_names[:20]}")
            print(f"[LoRA Lab] Available modules (first 20): {module_names[:20]}")
            
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "not authorized" in str(e).lower() or "login" in str(e).lower():
                log_msgs.append("Authentication error: Please check your Hugging Face token and model access permissions.")
                return 0.0, "\n".join(log_msgs), None
            else:
                log_msgs.append(f"Model loading error: {e}")
                print(f"[LoRA Lab] Model loading error: {e}")
                return 0.0, "\n".join(log_msgs), None
        log_msgs.append("Setting up LoRA config...")
        lora_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=32,  # Increased from 16
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        log_msgs.append("Tokenizing dataset...")
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
        ds_tok = ds.map(tokenize_fn, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        log_msgs.append("Starting training...")
        output_dir = save_path_val if save_path_val else os.path.join(os.getcwd(), "lora_adapter")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            log_msgs.append(f"Error creating directory: {e}")
            print(f"[LoRA Lab] Error creating directory: {e}")
            return 0.0, "\n".join(log_msgs), None
        # Try the whole training block, fallback if bf16/gpu error
        def train_block(fp16_val, bf16_val):
            args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,  # Match Colab batch size
                num_train_epochs=int(epochs),
                learning_rate=4e-5,  # Match Colab learning rate
                weight_decay=0.01,  # Match Colab weight decay
                max_grad_norm=1.0,  # Prevent exploding gradients
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                fp16=fp16_val,
                bf16=bf16_val,
                disable_tqdm=False,
                dataloader_pin_memory=False,  # Disable for MPS
                gradient_accumulation_steps=4,  # Effective batch size = 4
            )
            def log_callback(logs):
                if "loss" in logs:
                    loss_val = logs['loss']
                    epoch_val = logs.get('epoch', '?')
                    step_val = logs.get('step', '?')
                    log_msgs.append(f"Epoch {epoch_val}, Step {step_val}: loss={loss_val:.6f}")
                    print(f"[LoRA Lab] Epoch {epoch_val}, Step {step_val}: loss={loss_val:.6f}")
                    progress((logs.get('epoch', 0) / float(epochs)))
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tok,
                data_collator=data_collator,
                callbacks=[],
            )
            trainer.train()
        try:
            train_block(fp16, bf16)
        except Exception as e:
            if any(x in str(e).lower() for x in ["bf16", "gpu", "bfloat16"]):
                log_msgs.append("bf16 training failed, retrying with float32 (no mixed precision)...")
                print("[LoRA Lab] bf16 training failed, retrying with float32 (no mixed precision)...")
                try:
                    train_block(False, False)
                    log_msgs.append("Successfully retrained with fp16=False, bf16=False.")
                    print("[LoRA Lab] Successfully retrained with fp16=False, bf16=False.")
                except Exception as e2:
                    log_msgs.append(f"Training error after fallback: {e2}")
                    print(f"[LoRA Lab] Training error after fallback: {e2}")
                    return 0.0, "\n".join(log_msgs), None
            else:
                log_msgs.append(f"Training error: {e}")
                print(f"[LoRA Lab] Training error: {e}")
                return 0.0, "\n".join(log_msgs), None
        log_msgs.append("Training complete! Saving LoRA adapter...")
        try:
            model.save_pretrained(output_dir)
            log_msgs.append(f"LoRA adapter saved to: {output_dir}")
            print(f"[LoRA Lab] LoRA adapter saved to: {output_dir}")
        except Exception as e:
            log_msgs.append(f"Error saving LoRA adapter: {e}")
            print(f"[LoRA Lab] Error saving LoRA adapter: {e}")
            return 0.0, "\n".join(log_msgs), None
        return 1.0, "\n".join(log_msgs), output_dir
    except Exception as e:
        log_msgs.append(f"Error: {e}")
        print(f"[LoRA Lab] Error: {e}")
        return 0.0, "\n".join(log_msgs), None

def mvp_ui():
    with gr.Blocks(title="LoRA Lab MVP") as demo:
        gr.Markdown("# LoRA Lab\nFine-tune LLMs with LoRA on your Mac (MVP)")
        with gr.Tab("Data & Model Setup"):
            with gr.Row():
                dataset = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False)
            dataset.upload(validate_csv, inputs=dataset, outputs=[dataset, dataset_status])
            
            # Only show models that are available locally
            available_models = [m[1] for m in BASE_MODELS if check_model_availability(m[0])]
            with gr.Row():
                # Set a default value if we have models available
                default_model = available_models[0] if available_models else None
                model = gr.Dropdown(
                    choices=available_models,
                    value=default_model,  # Set default value
                    label="Select Base Model (Local Models Only)"
                )
                model_status = gr.Textbox(
                    label="Model Selection",
                    value=f"Selected: {default_model}" if default_model else "No models available.",
                    interactive=False
                )
                
            if not available_models:
                gr.Markdown("""⚠️ No local models found. You need to download models first:
                1. Run with internet connection once to download
                2. Or manually download and place in the 'models' directory""")
            
            def on_model_select(choice):
                return f"Selected: {choice}" if choice else "No model selected."
            model.change(on_model_select, inputs=model, outputs=model_status)
            
            with gr.Row():
                hf_token = gr.Textbox(label="Hugging Face Access Token (if required)", type="password", placeholder="Paste your token here if needed...")
            gr.Markdown("Need a token? [Get one here](https://huggingface.co/settings/tokens)")
        
        # --- Training Tab ---
        with gr.Tab("Training"):
            gr.Markdown("## Training Configuration")
            with gr.Row():
                lora_rank = gr.Slider(minimum=2, maximum=32, value=8, step=2, label="LoRA Rank")
                epochs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Epochs")
                learning_rate = gr.Number(value=5e-5, label="Learning Rate")
            save_path = gr.Textbox(label="Save LoRA Adapter To (directory)", placeholder="/Users/yourname/Documents/lora_adapter")
            confirm_save_path = gr.Button("Confirm Save Path")
            start_btn = gr.Button("Start Training")
            cancel_btn = gr.Button("Cancel Training", interactive=False)
            progress = gr.Progress()
            log = gr.Textbox(label="Training Log", lines=8, interactive=False)
            adapter_path = gr.Textbox(label="Adapter Path", visible=False)

            # State variables
            dataset_state = gr.State(None)
            model_state = gr.State(default_model)  # Initialize with default model if available
            save_path_state = gr.State("")

            def update_dataset_state(f, ds, m, sp):
                ds = f
                return ds, gr.update(interactive=bool(ds and m and sp))

            def update_model_state(m, ds, sp):
                return m, gr.update(interactive=bool(ds and m and sp))

            def update_save_path_state(sp, ds, m):
                return sp, gr.update(interactive=bool(ds and m and sp))

            dataset.upload(update_dataset_state, inputs=[dataset, dataset_state, model_state, save_path_state], outputs=[dataset_state, start_btn])
            model.change(update_model_state, inputs=[model, dataset_state, save_path_state], outputs=[model_state, start_btn])
            confirm_save_path.click(update_save_path_state, inputs=[save_path, dataset_state, model_state], outputs=[save_path_state, start_btn])

            def start_training(dataset_file, model_choice, lora_rank, epochs, learning_rate, hf_token, save_path_val):
                _, logs, adapter_dir = lora_train(dataset_file, model_choice, lora_rank, epochs, learning_rate, hf_token, save_path_val, progress)
                if adapter_dir:
                    return logs, gr.update(value=adapter_dir, visible=True)
                return logs, gr.update(value="", visible=False)
            start_btn.click(start_training, inputs=[dataset_state, model_state, lora_rank, epochs, learning_rate, hf_token, save_path_state], outputs=[log, adapter_path])

        # --- Playground Tab ---
        with gr.Tab("Playground"):
            gr.Markdown("## Test Your Model")
            with gr.Row():
                playground_model = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Base Model"
                )
            with gr.Row():
                playground_adapter = gr.Textbox(
                    label="Current LoRA Adapter Path",
                    placeholder="No LoRA adapter loaded",
                    value="",
                    interactive=True  # Make it interactive so users can paste paths
                )
                load_adapter_btn = gr.Button("Load LoRA From Path")
                use_trained_adapter_btn = gr.Button("Use Last Trained Adapter")
                clear_adapter_btn = gr.Button("Clear LoRA (Use Base Model)")
            
            # Add a status message for LoRA loading
            lora_status = gr.Textbox(
                label="LoRA Status",
                value="Using base model only",
                interactive=False
            )
            
            playground_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Type your prompt here...",
                lines=3
            )
            generate_btn = gr.Button("Generate")
            playground_output = gr.Textbox(label="Model Output", lines=4, interactive=False)

            # Enable Generate button if model is selected
            def enable_generate(model_choice, adapter_dir=None):
                if not model_choice:
                    return gr.update(interactive=False)
                return gr.update(interactive=True)
            
            playground_model.change(
                enable_generate,
                inputs=[playground_model, playground_adapter],
                outputs=generate_btn
            )

            # Function to validate and load LoRA adapter
            def load_lora_adapter(adapter_path, current_model):
                if not adapter_path:
                    return (
                        "No adapter path provided",
                        gr.update(interactive=True)
                    )
                
                # Check if this looks like a valid LoRA adapter
                required_files = ["adapter_config.json", "adapter_model.safetensors"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(adapter_path, f))]
                
                if missing_files:
                    return (
                        f"Invalid LoRA adapter: Missing {', '.join(missing_files)}",
                        gr.update(interactive=True)
                    )
                
                try:
                    # Try to load the adapter config to verify compatibility
                    import json
                    config_path = os.path.join(adapter_path, "adapter_config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # You could add more validation here based on the config
                    
                    return (
                        f"LoRA adapter loaded successfully from {os.path.basename(adapter_path)}",
                        gr.update(interactive=True)
                    )
                except Exception as e:
                    return (
                        f"Error loading LoRA adapter: {str(e)}",
                        gr.update(interactive=True)
                    )

            # Function to use the last trained adapter
            def use_last_trained(adapter_dir, model_choice):
                if not adapter_dir:
                    return (
                        gr.update(),
                        gr.update(),
                        "No recently trained adapter available",
                        gr.update(interactive=True)
                    )
                return (
                    gr.update(value=adapter_dir),
                    gr.update(value=model_choice),
                    f"Loaded recently trained adapter from {os.path.basename(adapter_dir)}",
                    gr.update(interactive=True)
                )
            
            # Function to clear LoRA and use base model
            def clear_lora():
                return (
                    gr.update(value=""),
                    "Using base model only",
                    gr.update(interactive=True)
                )
            
            # Connect the buttons
            load_adapter_btn.click(
                load_lora_adapter,
                inputs=[playground_adapter, playground_model],
                outputs=[lora_status, generate_btn]
            )
            
            use_trained_adapter_btn.click(
                use_last_trained,
                inputs=[adapter_path, model],
                outputs=[playground_adapter, playground_model, lora_status, generate_btn]
            )
            
            clear_adapter_btn.click(
                clear_lora,
                outputs=[playground_adapter, lora_status, generate_btn]
            )

            def run_inference(prompt, model_choice, adapter_dir=None, hf_token=None):
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                from peft import PeftModel
                
                if not prompt.strip():
                    return "Please enter a prompt."
                
                model_path = MODEL_ID_MAP[model_choice]
                if not check_model_availability(model_path):
                    return f"Error: Model not available locally at {model_path}"
                
                # Device selection
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                print(f"[LoRA Lab] Inference using device: {device}")
                
                try:
                    print(f"[LoRA Lab] Loading base model: {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                        attn_implementation="eager" if "gemma" in model_path.lower() else None
                    )
                    print(f"[LoRA Lab] Base model loaded successfully")
                    
                    # Load LoRA adapter if provided and exists
                    if adapter_dir and os.path.exists(adapter_dir):
                        print(f"[LoRA Lab] Loading LoRA adapter from: {adapter_dir}")
                        model = PeftModel.from_pretrained(base_model, adapter_dir)
                        print(f"[LoRA Lab] LoRA adapter loaded successfully")
                    else:
                        model = base_model
                        print("[LoRA Lab] Using base model without LoRA")
                    
                    model.eval()
                    model.to(device)
                    
                    # Format prompt for instruction models
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n"
                    
                    # Generate with proper settings
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1 if device=="cpu" else 0,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    output = pipe(formatted_prompt)
                    
                    # Extract the assistant's response
                    response = output[0]['generated_text']
                    if "<start_of_turn>assistant\n" in response:
                        response = response.split("<start_of_turn>assistant\n")[1]
                    if "<end_of_turn>" in response:
                        response = response.split("<end_of_turn>")[0]
                    
                    return response.strip()
                    
                except Exception as e:
                    print(f"[LoRA Lab] Inference error: {e}")
                    return f"Error during inference: {str(e)}"

            generate_btn.click(
                run_inference,
                inputs=[playground_prompt, playground_model, playground_adapter, hf_token],
                outputs=playground_output
            )
        gr.Markdown("---\nMVP: Dataset upload, model selection, and training config UI.")
    return demo

if __name__ == "__main__":
    app = mvp_ui()
    app.launch() 