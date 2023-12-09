import requests
from io import BytesIO
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import extract_features
import numpy as np
from PIL import Image

chroma_client = None
collection = None


# Initialize ChromaDB client and collection
def appstart():
    global chroma_client, collection
    # chroma_client = chromadb.HttpClient(host='localhost', port='8000')
    chroma_client = chromadb.Client()

    collection = chroma_client.create_collection(name="luxora")


appstart()


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None


def store_image_embedding_from_url(image_url):
    image = download_image(image_url)
    if image is None:
        raise ValueError("Could not download image from URL")

    embedding = extract_features(image).tolist()
    collection.add(
        embeddings=[embedding],
        documents=[image_url],
        ids=[image_url]
    )


def compare_embedding_with_url(current_url, before_urls):
    current_image = download_image(current_url)
    if current_image is None:
        raise ValueError("Could not download current image from URL")

    current_embedding = extract_features(current_image).tolist()
    similarities = []

    for url in before_urls:
        before_image = download_image(url)
        if before_image is None:
            continue

        before_embedding = extract_features(before_image).tolist()
        similarity = cosine_similarity(np.array(current_embedding).reshape(1, -1), np.array(before_embedding).reshape(1, -1))
        similarities.append((url, similarity[0][0]))

    return similarities
