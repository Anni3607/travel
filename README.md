# India Travel Recommender System

This is a basic content-based travel recommender system built using Python and Streamlit. It helps users discover destinations in India based on their interests and budget.

## Overview

The system uses natural language input from the user (e.g., "history, shopping") to recommend travel destinations with optimized itineraries. It applies basic machine learning techniques (TF-IDF vectorization and cosine similarity) to compare user preferences against a dataset of Indian destinations.

## Features

- Recommend destinations based on user input
- Filter trips by estimated budget
- Show detailed morning, afternoon, and evening itineraries
- Intuitive web-based interface using Streamlit

## How It Works

1. The user enters their travel interests (like "culture", "adventure", or "shopping").
2. The app vectorizes those interests using TF-IDF.
3. It calculates cosine similarity between the user input and the travel dataset.
4. Top matching destinations under the user's budget are displayed.

## Dataset

The dataset contains:
- Indian destinations
- Duration of trip (in days)
- Associated interests (up to 5 per destination)
- Suggested itinerary for each day (morning, afternoon, evening)

link : https://travel-dxuch9z435stybz9wocdty.streamlit.app/#agra-2-days
