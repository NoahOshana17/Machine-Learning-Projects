import unittest
from src.model import download_model
from src.pipeline import setup_llm

class TestModelAndPipeline(unittest.TestCase):

    def test_download_model(self):
        model_path = download_model("metaresearch/llama-2/pyTorch/7b-chat-hf")
        self.assertTrue(os.path.exists(model_path), "Model should be downloaded to the specified path.")

    def test_setup_llm(self):
        # Mock or actual setup code
        query_pipeline = setup_llm()
        self.assertIsNotNone(query_pipeline, "LLM should be set up properly.")

if __name__ == "__main__":
    unittest.main()