import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from capybaradb.model import EmbeddingModel


class TestEmbeddingModel:
    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embedding_model_init_float32(self, mock_model, mock_tokenizer):
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = EmbeddingModel(precision="float32", device="cpu")

        assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.device == "cpu"
        assert model.precision == "float32"
        mock_tokenizer.from_pretrained.assert_called_once_with(model.model_name)
        mock_model.from_pretrained.assert_called_once_with(
            model.model_name, torch_dtype=torch.float32
        )

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embedding_model_init_float16(self, mock_model, mock_tokenizer):
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = EmbeddingModel(precision="float16", device="cuda")

        assert model.precision == "float16"
        assert model.device == "cuda"
        mock_model.from_pretrained.assert_called_once_with(
            model.model_name, torch_dtype=torch.float16
        )

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embedding_model_init_binary(self, mock_model, mock_tokenizer):
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        model = EmbeddingModel(precision="binary", device="cpu")

        assert model.precision == "binary"
        mock_model.from_pretrained.assert_called_once_with(model.model_name)

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embed_single_string(self, mock_model, mock_tokenizer, sample_text):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel()
        result = model.embed(sample_text)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
        mock_tokenizer_instance.assert_called_once()

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embed_list_of_strings(self, mock_model, mock_tokenizer, sample_documents):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(3, 3, 384)]

        model = EmbeddingModel()
        result = model.embed(sample_documents)

        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3
        mock_tokenizer_instance.assert_called_once()

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embed_binary_precision(self, mock_model, mock_tokenizer, sample_text):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel(precision="binary")
        result = model.embed(sample_text)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert torch.all((result == 0) | (result == 1))

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_embed_float16_precision(self, mock_model, mock_tokenizer, sample_text):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel(precision="float16")
        result = model.embed(sample_text)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float16

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_search_basic(self, mock_model, mock_tokenizer, sample_documents):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel()

        embeddings = torch.randn(3, 384)
        query = "test query"

        with patch.object(
            model, "embed", return_value=torch.randn(1, 384)
        ) as mock_embed:
            indices, scores = model.search(query, embeddings, top_k=2)

            assert isinstance(indices, torch.Tensor)
            assert isinstance(scores, torch.Tensor)
            assert len(indices) == 2
            assert len(scores) == 2
            mock_embed.assert_called_once_with(query)

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_search_binary_precision(self, mock_model, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel(precision="binary")

        embeddings = torch.randn(3, 384)
        query = "test query"

        with patch.object(
            model, "embed", return_value=torch.randn(1, 384)
        ) as mock_embed:
            indices, scores = model.search(query, embeddings, top_k=2)

            assert isinstance(indices, torch.Tensor)
            assert isinstance(scores, torch.Tensor)
            mock_embed.assert_called_once_with(query)

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_search_top_k_larger_than_embeddings(self, mock_model, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        model = EmbeddingModel()

        embeddings = torch.randn(2, 384)
        query = "test query"

        with patch.object(
            model, "embed", return_value=torch.randn(1, 384)
        ) as mock_embed:
            indices, scores = model.search(query, embeddings, top_k=5)

            assert len(indices) == 2
            assert len(scores) == 2

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_mean_pooling(self, mock_model, mock_tokenizer):
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = EmbeddingModel()

        token_embeddings = torch.randn(2, 5, 384)  # batch_size, seq_len, hidden_size
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

        result = model._mean_pooling((token_embeddings,), attention_mask)

        assert result.shape == (2, 384)
        assert isinstance(result, torch.Tensor)

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_mean_pooling_zero_attention(self, mock_model, mock_tokenizer):
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = EmbeddingModel()

        token_embeddings = torch.randn(1, 3, 384)
        attention_mask = torch.tensor([[0, 0, 0]])  # All zeros

        result = model._mean_pooling((token_embeddings,), attention_mask)

        assert result.shape == (1, 384)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    @patch("capybaradb.model.AutoTokenizer")
    @patch("capybaradb.model.AutoModel")
    def test_device_handling(self, mock_model, mock_tokenizer):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        model = EmbeddingModel(device="cuda")

        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model_instance.return_value = [torch.randn(1, 3, 384)]

        with patch("torch.cuda.is_available", return_value=True):
            result = model.embed("test")

            assert isinstance(result, torch.Tensor)
            mock_tokenizer_instance.assert_called_once()
