class LLMHandler:
    llm_backend = "mlx"

    def initialize(self, checkpoint_dir, lm_model_path, backend="vllm", device="auto"):
        return f"✅ {lm_model_path}", True
