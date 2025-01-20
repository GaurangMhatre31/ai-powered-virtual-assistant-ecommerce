import spacy
from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from flask_cors import CORS  # Importing CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire application or specific routes
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:3000"}})

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Sample FAQ data
faq_data = [
    {"question": "What is the return policy?", "answer": "You can return items within 30 days."},
    {"question": "How do I track my order?", "answer": "You can track your order by logging into your account."},
    {"question": "What payment methods are accepted?", "answer": "We accept credit cards, PayPal, and more."}
]

# Sample data for training intents
data = {
    'text': [
        'I am looking for a wireless headphone under $100.',
        'Where is my order #12345?',
        'What is your return policy?',
        'Show me some product recommendations.',
        'What is the status of my order?'
    ],
    'intent': [
        'product_search',
        'order_tracking',
        'faq_inquiry',
        'product_search',
        'order_tracking'
    ]
}

# Create the vectorizer and classifier pipeline
vectorizer = TfidfVectorizer()
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Train the intent classification model
X_train = data['text']
y_train = data['intent']
model.fit(X_train, y_train)

# Function to predict intent
def predict_intent(text):
    return model.predict([text])[0]

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function for FAQ retrieval using cosine similarity
def retrieve_answer(query):
    faq_texts = [faq["question"] for faq in faq_data]
    faq_matrix = vectorizer.fit_transform(faq_texts)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, faq_matrix)
    best_match_idx = similarities.argmax()
    return faq_data[best_match_idx]["answer"]

# Greeting function based on time of day
def get_greeting():
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        return "Good Morning!"
    elif 12 <= current_hour < 18:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

# Flask route for chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("user_input")
    
    # Greet the user based on the time of day
    greeting = get_greeting()
    
    # Handle different user intents
    intent = predict_intent(user_input)
    response = ""

    if intent == "product_search":
        response = "Here are some wireless headphones under $100 that you might like: [Product 1, Product 2, Product 3]."
    
    elif intent == "order_tracking":
        # Extract order ID and simulate order tracking
        order_id = extract_entities(user_input)
        response = f"Your order #{order_id} is scheduled to be delivered in 3 days."
    
    elif intent == "faq_inquiry":
        # Retrieve FAQ answer from knowledge base
        response = retrieve_answer(user_input)
    
    else:
        response = "I'm sorry, I didn't understand that."

    # Final response with greeting
    return jsonify({"response": f"{greeting} How can I assist you? {response}"})

# Main entry point to start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
