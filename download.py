from huggingface_hub import snapshot_download
from config import Config

snapshot_download(
    repo_id="meta-llama/Llama-3.1-70B-Instruct", 
    use_auth_token=Config.AIHF_KEY, 
    local_dir="/home/ubuntu/marketxmind/Llama-3_1_70B_Instruct" 
)
