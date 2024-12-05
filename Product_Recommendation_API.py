from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random 

# FastAPI instance
app = FastAPI()

# Load your datasets
whole_sale_company = pd.read_csv('wholesale_companies.csv')
whole_sale_customer = pd.read_csv('wholesale_customers.csv')
whole_sale_transactions = pd.read_csv('wholesale_transactions.csv')

# Example list of items
item_names = [
    "Organic Apples",
    "Whole Grain Bread",
    "Almond Milk",
    "Free-Range Eggs",
    "Grass-Fed Beef",
    "Quinoa",
    "Fresh Spinach",
    "Avocado Oil",
    "Dark Chocolate",
    "Greek Yogurt",
    "Oatmeal",
    "Raw Honey",
    "Fresh Salmon",
    "Coconut Water",
    "Kale Chips",
    "Protein Powder",
    "Mixed Nuts",
    "Basmati Rice",
    "Spaghetti Pasta",
    "Extra Virgin Olive Oil"
]

# Randomize item names in transactions
whole_sale_transactions['item_name'] = [random.choice(item_names) for _ in range(len(whole_sale_transactions))]

# Merge and preprocess data
merged_data = whole_sale_transactions.merge(whole_sale_company, on='company_id').merge(whole_sale_customer, on='customer_id')
merged_data = merged_data.rename(columns={
    'company_id_x': 'company_id',
    'name_x': 'company_name',
    'location_x': 'company_location',
    'name_y': 'customer_name',
    'location_y': 'customer_location'
})

# Create customer profiles
customer_profiles = merged_data[['customer_id', 'item_name', 'quantity', 'total_amount', 'inventory_value', 'business_type', 'credit_limit', 'customer_location']]

# Create item matrix
customer_item_matrix = customer_profiles.pivot_table(index='customer_id', columns='item_name', values='quantity', fill_value=0)

# Fixing the SettingWithCopyWarning by creating a copy
customer_profiles = customer_profiles.copy()

# Generate 'item_string' for each customer
customer_profiles['item_string'] = customer_profiles.groupby('customer_id')['item_name'].transform(lambda x: ' '.join(x))

# Drop duplicates
customer_profiles = customer_profiles[['customer_id', 'item_string']].drop_duplicates()

# Generate similarity matrix
vectorizer = TfidfVectorizer()
item_vectors = vectorizer.fit_transform(customer_profiles['item_string'])
similarity_matrix = cosine_similarity(item_vectors, item_vectors)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_profiles['customer_id'], columns=customer_profiles['customer_id'])

# Pydantic model for input validation
class RecommendationRequest(BaseModel):
    customer_id: int
    top_n: Optional[int] = 5
    min_inventory: Optional[int] = 0

@app.get("/")
def root():
    return {"message": "Welcome to the Wholesale Recommendation API!"}

@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest):
    customer_id = request.customer_id
    top_n = request.top_n
    min_inventory = request.min_inventory

    # Validate customer_id
    if customer_id not in similarity_df.index:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found.")

    # Get similar customers and recommendations
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).index[1:top_n+1]
    recommended_items = customer_item_matrix.loc[similar_customers].mean(axis=0).sort_values(ascending=False)

    # Filter items by inventory
    available_items = merged_data[merged_data['inventory_value'] >= min_inventory]['item_name'].unique()
    filtered_recommendations = [item for item in recommended_items.index if item in available_items]
    top_recommendations = filtered_recommendations[:top_n]

    return {
        "customer_id": customer_id,
        "recommendations": top_recommendations
    }
