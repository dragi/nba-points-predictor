"""Streamlit UI for predicting an NBA player's points in an upcoming game."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import (
    available_opponents,
    available_players,
    load_bundle,
    load_history,
    predict_points,
)


@st.cache_resource
def get_model():
    """Load the trained model bundle once per session."""
    return load_bundle()


@st.cache_data
def get_history():
    """Load the game history table once per session."""
    return load_history()


def main():
    """Render the prediction form and show the model's output."""
    st.set_page_config(page_title="NBA Points Predictor", page_icon="🏀")
    st.title("🏀 NBA Player Points Predictor")
    st.caption("Predicts a player's points for their next game from recent form, "
               "opponent defense, rest, and home court.")

    history = get_history()
    bundle = get_model()

    player = st.selectbox("Player", available_players(history))
    opponent = st.selectbox("Opponent", available_opponents(history))
    venue = st.radio("Venue", ["Home", "Away"], horizontal=True)
    rest_days = st.slider("Days of rest", min_value=0, max_value=7, value=2)

    if st.button("Predict points", type="primary"):
        points = predict_points(
            player,
            opponent,
            is_home=venue == "Home",
            rest_days=rest_days,
            history=history,
            bundle=bundle,
        )
        st.metric(f"{player} vs {opponent}", f"{points:.1f} pts")
        if rest_days <= 1:
            st.caption("Back-to-back game.")


if __name__ == "__main__":
    main()
