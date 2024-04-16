import yaml

yaml_config = """
base_model: openaccess-ai-collective/tiny-mistral
gate_mode: hidden
dtype: bfloat16
experts:
  - source_model: openaccess-ai-collective/tiny-mistral
    positive_prompts:
      - "math"
    # You can add negative_prompts if needed
  - source_model: openaccess-ai-collective/tiny-mistral

    positive_prompts:
      - "science"
  - source_model: openaccess-ai-collective/tiny-mistral
    positive_prompts:
      - "writing"
    # You can add negative_prompts if needed
  - source_model: openaccess-ai-collective/tiny-mistral
    positive_prompts:
      - "general"
"""

config = yaml.safe_load(yaml_config)

runtime = "CPU" # @param ["CPU", "CPU + High-RAM", "GPU"]
branch = "mixtral" # @param ["main", "mixtral"]
trust_remote_code = False # @param {type:"boolean"}

# Save config as yaml file
with open('config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(config, f)

# Base CLI
if branch == "main":
    cli = f"mergekit-yaml {config} merge --copy-tokenizer"
elif branch == "mixtral":
    cli = f"mergekit-moe {config} merge --copy-tokenizer --i-understand-this-is-not-useful-without-training"

# Additional arguments
if runtime == "CPU":
    cli += " --allow-crimes --out-shard-size 1B --lazy-unpickle"
elif runtime == "GPU":
    cli += " --cuda --low-cpu-memory"
if trust_remote_code:
    cli += " --trust-remote-code"

print(cli)

# Merge models
{cli}