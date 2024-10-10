# Near Earth Objects

This application predicts whether near-Earth objects (NEOs) pose a hazard. It utilizes various models, each with its own accuracy score, to assess the potential threat.

## Logistic Regression

The initial model employed by this application is a logistic regression , which achieves an approximate accuracy of 85%.

## Support Vector Machine

Additionally, the application uses a Support Vector Machine (SVM) model, specifically employing the Gaussian kernel, which has an approximate accuracy of 88%.


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