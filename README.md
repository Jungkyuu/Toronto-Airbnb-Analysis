# Toronto Airbnb Data Analysis

This project analyzes Airbnb listings data from Toronto using Python, with a focus on data cleaning, visualization, sentiment analysis, and machine learning-based price prediction.

## 📊 Features

- **Data Preprocessing**: Cleaned and filtered raw listing data.
- **Visualization**:
  - Average price by room type (bar chart).
  - Average price by neighbourhood (interactive Folium map).
  - Sentiment score vs price (scatter plot).
- **Sentiment Analysis**: Applied VADER sentiment analysis on listing titles (`name` column).
- **Machine Learning**:
  - Linear regression model to predict price based on sentiment scores.
  - Output includes RMSE and prediction samples.

## 🧰 Technologies Used

- Python 3
- pandas, numpy
- matplotlib, seaborn
- folium (for map rendering)
- nltk (VADER sentiment analysis)
- scikit-learn (for linear regression)

## 🗂️ File Structure

| File | Description |
|------|-------------|
| `Toronto_Airbnb_Analysis.py` | Main Python script with data analysis and ML pipeline |
| `room_type_price.png` | Bar chart showing avg price by room type |
| `sentiment_vs_price.png` | Scatter plot of sentiment score vs price |
| `toronto_airbnb_map.html` | Interactive map of average prices by neighbourhood |
| `listings.csv` | Raw data file from Inside Airbnb *(Included in repo)* |

## 📥 How to Use

1. **Download dataset**  
   Visit [Inside Airbnb - Toronto](http://insideairbnb.com/get-the-data.html) and download the `listings.csv` file.

2. **Run the analysis**

```bash
python3 Toronto_Airbnb_Analysis.py
