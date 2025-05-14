# Toronto Airbnb Data Analysis

This project explores Airbnb listings in Toronto using Python.  
The goal is to understand pricing patterns and how listing titles might reflect value.

## 🔍 What’s Included

- Cleaned the raw Airbnb data and handled missing values
- Created visualizations:
  - Bar chart of average prices by room type
  - Interactive map showing average prices by neighbourhood
  - Scatter plot comparing listing title sentiment and price
- Built a simple linear regression model to check if sentiment affects pricing

## 📁 Files

| File | Description |
|------|-------------|
| `Toronto_Airbnb_Analysis.py` | Main script: data prep, visualization, and regression |
| `room_type_price.png` | Average price by room type (bar chart) |
| `sentiment_vs_price.png` | Sentiment score vs price (scatter plot) |
| `toronto_airbnb_map.html` | Interactive price map by neighbourhood |
| `listings.csv` | Source data (not included in repo, see below) |

## 📥 How to Run

1. Download the dataset  
   [Inside Airbnb - Toronto](http://insideairbnb.com/get-the-data.html)

2. Run the analysis

```bash
python3 Toronto_Airbnb_Analysis.py
