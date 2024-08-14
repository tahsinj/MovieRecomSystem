# Movie Recommendation System
## Overview
This project is a movie recommendation system developed using TensorFlow, leveraging collaborative filtering and content-based filtering techniques. It processes the MovieLens dataset to create user-movie interaction matrices and utilizes advanced machine learning algorithms to predict and recommend movies based on user preferences.
The Movie Recommendation System is designed to provide personalized movie suggestions to users. By analyzing the MovieLens dataset, the system can predict user preferences and recommend movies that users are likely to enjoy. This project showcases the integration of collaborative filtering and content-based filtering techniques to deliver accurate and relevant recommendations.

## Features
- Predicts and recommends movies based on user preferences.
- Utilizes collaborative filtering and content-based filtering techniques.
- Processes the MovieLens dataset to create user-movie interaction matrices.
- Implements mean normalization to enhance recommendation accuracy.
- Developed using TensorFlow and Python.

## Dataset
The project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/), which contains movie ratings and metadata. This project specifically uses the MovieLens Latest Small Dataset recommended for education and development.

## Technologies Used
- TensorFlow
- Python
- Pandas
- NumPy
- Scikit-Learn

## Getting Started
1. Clone the Repository:
```bash
git clone https://github.com/tahsinj/MovieRecomSystem.git
```
2. Create a Virtual Environment:
```bash
python -m venv .venv
.venv/Scripts/activate  # On MacOS/Linux use `source venv/bin/activate`
```
3. Install Dependencies:
```bash
pip install requirements.txt
```
4. Run the CLI Program:
```bash
python src/main.py
```

## Citations
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
