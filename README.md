# Near Earth Objects

This application predicts whether near-Earth objects (NEOs) pose a hazard. It utilizes various models, each with its own accuracy score, to assess the potential threat.

## Models

This application uses an AdaBoost model, trained over 4 different databases with different features each, you can try and predict with each database.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://irisclassification-qmrjl8zkbypcggcqmxewjz.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```