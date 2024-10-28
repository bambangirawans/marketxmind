import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

model_path = r"/home/ubuntu/marketxmind/Llama-3_2_1B_Instruct/"

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16  # Adjust to torch.float16 if needed
    )

# Tie the model weights
model.tie_weights()

# Load the model with offloading and device mapping
model = load_checkpoint_and_dispatch(
    model,
    model_path,
    device_map="auto",  
    offload_folder=r"/home/ubuntu/marketxmind/offload/"  
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id explicitly
)

# Define conversation structure
conversation = [
    {"role": "system", "content": "Hi, I am a market-X-mind chatbot who is a business analyst specializing in customer loyalty and engagement strategies."},
]

# Simulate user input
user_input = "hi"
if user_input:
    conversation.append({"role": "user", "content": user_input})
    
    try:
        # Convert conversation to a single prompt text for the pipeline
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        # Generate response
        response = pipe(prompt_text, max_new_tokens=500)
        assistant_response = response[0]['generated_text'].strip()
        
        # Append assistant's response to conversation
        conversation.append({"role": "assistant", "content": assistant_response})
        
        # Print assistant response
        print(assistant_response)
        
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Please enter a valid message.")

