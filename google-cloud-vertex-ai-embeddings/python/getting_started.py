import vertexai

from vertexai.language_models import TextEmbeddingModel
import vertexai
vertexai.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='text-embeddings',
)
def text_embedding() -> list:
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(["What is life?"])
    for embedding in embeddings:
        vector = embedding.values
        print(f"Length of Embedding Vector: {len(vector)}")
    return vector


if __name__ == "__main__":
    text_embedding()
