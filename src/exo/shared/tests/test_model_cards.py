from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.types.models import ModelId

def test_qwen3_235b_model_id():
    """Verify the model ID for Qwen3 235B 6-bit is correct."""
    card = MODEL_CARDS["qwen3-235b-a22b-6bit"]
    assert card.model_id == "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
    assert card.metadata.model_id == "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
    assert card.metadata.max_sequence_length == 262144
