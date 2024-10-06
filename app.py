import json
from flask import Flask, request, jsonify
from mistralai.client import MistralClient
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from mistralai.models.chat_completion import ChatMessage
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/recipe": {"origins": "http://localhost:3000"}})
# ===========================
# Initialization and Setup
# ===========================

# API and model setup
api_key = "uAyQH2qjnN5Batgf1YDp24KB3BLKLHBK"
model = "open-mistral-7b"
mistral_client = MistralClient(api_key=api_key)

# Encoder and Qdrant client setup
encoder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant_client = QdrantClient(":memory:")

# Load documents from a JSON file
with open('products_with_recipes_and_ingredients_test.json', 'r', encoding='utf-8') as file:
    documents = json.load(file)

collection_name = "my_recipes"

# Define the vector configuration for Qdrant
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

# Upload documents to Qdrant using ingredients for vectorization
qdrant_client.upload_points(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(", ".join(doc["Ingredients"])).tolist(),
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

# ===========================
# Helper Functions
# ===========================

def retrieve_recipes(query):
    """
    Retrieve recipes from Qdrant based on the user's ingredient query.
    """
    query_vector = encoder.encode(query).tolist()
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3  # Adjust the limit based on preference
    )
    return search_results

def generate_recipe_response(user_ingredients, search_results):
    """
    Generate a recipe suggestion using Mistral AI based on the search results.
    """
    user_input=user_ingredients
    if not search_results:
        return "I couldn't find any recipes matching your ingredients."

    context = "\n\n".join([
        f"Product: {hit.payload['Product']}\n"
        f"Details: {hit.payload['Details']}\n"
        f"Ingredients: {', '.join(hit.payload['Ingredients'])}\n"
        f"Recipe: {hit.payload['Recipe']}"
        for hit in search_results
    ])

    # Prepare the messages for Mistral AI
    final_prompt=(f"you represent National foods.if the user message is only about greetings respond him in that manner.If user asks about any thing related to food or recipe creation use the context below to answer him in that manner "
        f"Regards should always be from national food bot"
        f"Context:\n{context}\n\nUser Query:\n{user_input}\n\nResponse:")

    # Call the Mistral client to generate a response
    chat_response = mistral_client.chat(
                model=model,
                messages=[
                    ChatMessage(role="system", content=final_prompt),
                    ChatMessage(role="user", content=user_input),
                    
                ]
            )
    
    return chat_response.choices[0].message.content if chat_response.choices else "No response from Mistral AI."

# ===========================
# Routes
# ===========================

@app.route('/recipe', methods=['POST'])
def recipe():
    """
    Endpoint to receive user ingredients and return recipe suggestions.
    """
    data = request.get_json()
    
    # Validate input
    if not data or 'ingredients' not in data:
        return jsonify({'error': 'Please provide ingredients in the request body.'}), 400
    
    user_ingredients = data['ingredients']
    
    # Retrieve relevant recipes
    search_results = retrieve_recipes(user_ingredients)
    
    # Generate AI response
    response = generate_recipe_response(user_ingredients, search_results)
    
    return jsonify({'response': response})

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint to check if the service is running.
    """
    return jsonify({'message': 'Recipe Suggestion API is running.'}), 200

# ===========================
# Run the Flask App
# ===========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)