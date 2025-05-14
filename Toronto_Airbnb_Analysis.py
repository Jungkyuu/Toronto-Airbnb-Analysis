import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------------------
# 1. Load the data
# ------------------------------
listings = pd.read_csv('listings.csv')

# ------------------------------
# 2. Data preprocessing
# ------------------------------
listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
listings = listings.dropna(subset=['price', 'neighbourhood', 'room_type'])

# ------------------------------
# 3. Average price by room type
# ------------------------------
plt.figure(figsize=(10,6))
sns.barplot(x='room_type', y='price', data=listings)
plt.title('Average Price by Room Type')
plt.ylabel('Average Price ($)')
plt.xlabel('Room Type')
plt.tight_layout()
plt.savefig('room_type_price.png')
plt.close()

# ------------------------------
# 4. Average price by neighbourhood on map
# ------------------------------
neighbourhood_price = listings.groupby('neighbourhood')['price'].mean().reset_index()
toronto_map = folium.Map(location=[43.6532, -79.3832], zoom_start=11)

for idx, row in neighbourhood_price.iterrows():
    lat = listings[listings['neighbourhood'] == row['neighbourhood']]['latitude'].mean()
    lon = listings[listings['neighbourhood'] == row['neighbourhood']]['longitude'].mean()
    if not np.isnan(lat) and not np.isnan(lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=f"{row['neighbourhood']}: ${row['price']:.2f}",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(toronto_map)

toronto_map.save("toronto_airbnb_map.html")

# ------------------------------
# 5. Sentiment analysis (using listing name)
# ------------------------------
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
listings['name'] = listings['name'].fillna('')
listings['sentiment'] = listings['name'].apply(lambda x: sid.polarity_scores(x)['compound'])

plt.figure(figsize=(10,6))
sns.scatterplot(x='sentiment', y='price', data=listings)
plt.title('Sentiment Score of Listing Name vs Price')
plt.xlabel('Sentiment Score')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.savefig('sentiment_vs_price.png')
plt.close()

# ------------------------------
# 6. Predicting price using sentiment score
# ------------------------------
listings_ml = listings.dropna(subset=['sentiment'])

if len(listings_ml) >= 10:
    X = listings_ml[['sentiment']]
    y = listings_ml['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n✅ Analysis complete!")
    print("📊 Generated: room_type_price.png, sentiment_vs_price.png")
    print("🗺️  Generated: toronto_airbnb_map.html")
    print("\n🧠 Machine Learning: Price Prediction")
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("Sample Predictions (first 5):", y_pred[:5])
else:
    print("❌ Not enough data to run machine learning model.")
