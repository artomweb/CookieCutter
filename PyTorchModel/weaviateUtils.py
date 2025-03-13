import weaviate
from weaviate.classes.config import Configure, Property, DataType
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

def compute_moments_and_hu(contour):
    """Calculate raw moments and Hu moments from a contour."""
    # Compute moments
    moments = cv2.moments(contour)
    print(f"Raw Moments: m00={moments['m00']:.2f}, m10={moments['m10']:.2f}, m01={moments['m01']:.2f}")

    # Compute Hu moments (7 invariant moments)
    hu_moments = cv2.HuMoments(moments).flatten()


    # Log-transform Hu moments to handle large ranges (common practice)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # Add small epsilon to avoid log(0)
    print(f"Hu Moments: {hu_moments}")
    
    # Convert NumPy array to Python list for Weaviate compatibility
    hu_moments_list = hu_moments.tolist()

    hu_moments[6] = abs(hu_moments[6])
    
    return moments, hu_moments_list

def query_weaviate_hu_moments(collection, hu_moments, threshold=.1):
    """Query Weaviate to check if similar Hu moments exist."""
    response = collection.query.near_vector(
        near_vector=hu_moments,
        distance=threshold,
        limit=1,
        # target_vector="hu_moments",
        return_properties=["image_path"],
        return_metadata=MetadataQuery(distance=True)
    )
    for o in response.objects:
        print(o.properties)
        print(o.metadata.distance)
    if response.objects:
        return response.objects[0].properties["image_path"]
    print("No similar vectors")
    return None

def getWeaviate(deleteOld = False):
    # Initialize Weaviate client
    client = weaviate.connect_to_local()

    collection_name = "cookieCutters"
    if deleteOld:
        client.collections.delete(collection_name)
    collection_exists = client.collections.exists(collection_name)


    if not collection_exists:
        print("Creating Collection")
        collection = client.collections.create(
                name=collection_name,
                properties=[
                    wvc.config.Property(name="image_path", data_type=wvc.config.DataType.TEXT, description="Path to the image file"),
                    wvc.config.Property(name="hu_moments", data_type=wvc.config.DataType.NUMBER_ARRAY, description="Hu Moments feature vector")
                ],
                vectorizer_config=[
                    wvc.config.Configure.NamedVectors.none(
                        name="hu_moments" , # Define hu_moments as a named vector with no vectorizer
                        vector_index_config=Configure.VectorIndex.hnsw()
                    )
                ],
            )
    else:
        collection = client.collections.get(collection_name)
    return client, collection