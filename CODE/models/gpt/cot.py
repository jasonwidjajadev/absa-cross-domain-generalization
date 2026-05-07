# COMP6713 26T1 NLP Project
# Group: Cold Tuna
# Chain of Thought (CoT) prompting for ABSA Task.
# This python file contains the code for the CoT method for our ABSA Task. When given a sentence and an aspect term,
# GPT-5.4-mini will be used to predict the sentiment polarity for the aspect term.
# A few-shot CoT prompt is used to mainly test this method.

import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report
from pathlib import Path

# Setup
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL  = "gpt-5.4-mini"

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent
COMBINED_TEST  = DATA_DIR / "MISC/data/combined/test.csv"
LAPTOP_TEST  = DATA_DIR / "MISC/data/semeval2014/laptop/processed/test.csv"

# Prompts
# Baseline
ZERO_SHOT = """
Given a sentence and a specific aspect term, classify the sentiment expressed toward that aspect.
You can only reply with a single word: positive, negative, or neutral.
""".strip()

# CoT few-shot prompt
FEW_SHOT = """
You are an expert in Aspect-Based Sentiment Analysis (ABSA).
Given a sentence and a specific aspect term, determine the sentiment expressed toward that aspect in the sentence.

Think and do this step by step and follow this procedure:
1. Locate the aspect term in the sentence.
2. Identify the words or phrases that describe or relate to it.
3. Decide whether the overall sentiment toward that aspect is positive, negative, or neutral.

You can only reply with a single word: positive, negative, or neutral.
""".strip()

# Restaurant few-shot example
# simulate chat with GPT-5.4-mini, user and expected response
R_FEW_SHOT = [
    {
        "role": "user",
        "content": 'Sentence: "The pasta was completely overcooked and bland."\nAspect: "pasta"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'pasta' is the subject. "
            "2: Find descriptors: 'completely overcooked' and 'bland' describe the pasta. "
            "3: Both descriptors show that the reviewer is unsatisfied with the food quality. "
            "negative")
    },
    {
        "role": "user",
        "content": 'Sentence: "Our waiter was attentive and checked on us several times throughout the meal."\nAspect: "waiter"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'waiter' is the subject. "
            "2: Find descriptors: 'attentive' and 'checked on us several times' both describe the waiter positively. "
            "3: Attentiveness and regular check-ins are valued service qualities. "
            "positive")
    },
    {
        "role": "user",
        "content": 'Sentence: "The prices are a bit high for what you get, but not outrageous."\nAspect: "prices"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'prices'. "
            "2: Find descriptors: 'a bit high for what you get' shows unsatisfied but 'not outrageous' makes it lighter. "
            "3: Overall prices are higher than expected. "
            "negative")
    },
    {
        "role": "user",
        "content": 'Sentence: "The ambience is nothing special... just a standard diner feel."\nAspect: "ambience"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'ambience'. "
            "2: Find descriptors: 'nothing special' and 'standard diner feel' shows a middle stance. "
            "3: The reviewer has no praise or complaint. "
            "neutral")
    },
]

# Laptop-domain few-shot examples (covers battery, screen, keyboard, performance)
L_FEW_SHOT = [
    {
        "role": "user",
        "content": 'Sentence: "The battery drains within two hours even on light use."\nAspect: "battery"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'battery'. "
            "2: Find descriptors: 'drains within two hours even on light use' shows very poor battery life. "
            "3: This is a clear complaint about the battery life of the laptop. "
            "negative"
        )
    },
    {
        "role": "user",
        "content": 'Sentence: "The display is absolutely stunning with vibrant colours and sharp text."\nAspect: "display"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'display'. "
            "2: Find descriptors: 'absolutely stunning', 'vibrant colours', and 'sharp text' all praise the screen. "
            "3: Strong positive feelings towards the display. "
            "positive"
        )
    },
    {
        "role": "user",
        "content": 'Sentence: "The keyboard is neither great nor terrible... just average for the price."\nAspect: "keyboard"'
    },
    {
        "role": "assistant",
        "content": (
            "1: Locate the aspect: 'keyboard'. "
            "2: Find descriptors: 'neither great nor terrible' and 'just average' express no strong take. "
            "3: The reviewer is in the middle on the keyboard quality. "
            "neutral"
        )
    },
]

# Load data
combined_df = pd.read_csv(COMBINED_TEST)
laptop_df = pd.read_csv(LAPTOP_TEST)
# exclude conflict polarity.
laptop_df = laptop_df[laptop_df["polarity"] != "conflict"].copy()


# GPT-5.4-mini prediction function for zero-shot using openAI API.
def predict_sentiment_zero_shot(sentence: str, aspect: str) -> str:
    """Zero-shot baseline prediction using GPT-5.4-mini.
    Given a sentence and an aspect, it predicts and returns the sentiment."""
    messages = [
        {"role": "system", "content": ZERO_SHOT},
        {"role": "user", "content": f'Sentence: "{sentence}"\nAspect: "{aspect}"'}
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0
    )
    reply = response.choices[0].message.content.strip().lower()
    sentiment = reply.split()[-1]
    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "error"
    return sentiment

# GPT-5.4-mini prediction function for few-shot CoT using openAI API.
# IMPORT THIS FUNCTION FOR USE IN GRADIO DEMO!
def predict_sentiment(sentence: str, aspect: str) -> str:
    """Few-shot CoT prediction using restaurant and laptop examples with GPT-5.4-mini.
    Given a sentence and an aspect with CoT prompting, it predicts and returns the sentiment."""
    messages = [
        {"role": "system", "content": FEW_SHOT},
        *R_FEW_SHOT,
        *L_FEW_SHOT,
        {"role": "user", "content": f'Sentence: "{sentence}"\nAspect: "{aspect}"'}
    ]
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0
    )
    reply = response.choices[0].message.content.strip().lower()
    sentiment = reply.split()[-1]
    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "error"
    return sentiment

# Run the prediction on the dataset and store results in a new df
def batch_evaluate(df: pd.DataFrame, mode: str = "few_shot_cot") -> pd.DataFrame:
    """
    Runs the sentiment prediction on every (sentence, aspect) row in the df.
    """
    results = []
    total = len(df)

    # loop through each row in the dataset and predict the sentiment
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["text"]
        aspect = row["target"]
        true_sentiment = row["polarity"].lower().strip()

        # use zero-shot or few-shot CoT prediction based on whats selected
        try:
            if mode == "few_shot_cot":
                pred = predict_sentiment(sentence, aspect)
            else:
                pred = predict_sentiment_zero_shot(sentence, aspect)
            time.sleep(1.0)
        except Exception as e:
            print("error")
            pred = "error"

        results.append({
            "text": sentence,
            "target": aspect,
            "polarity": true_sentiment,
            "predicted": pred,
            "correct": pred == true_sentiment
        })
        # every 20 print current progress
        if idx % 20 == 0:
            print(f"[{idx}/{total}] running accuracy: {sum(r['correct'] for r in results) / len(results):.3f}")
    return pd.DataFrame(results)

# Evaluation function that prints out metrics
def evaluate(results_df: pd.DataFrame) -> None:
    """print the evaluation metrics from results."""
    valid = results_df[results_df["predicted"] != "error"]
    print("Results Of Experiment:")

    print("\nClassification Report:")
    print(classification_report(
        valid["polarity"],
        valid["predicted"],
        labels=["positive", "negative", "neutral"],
        zero_division=0
    ))

if __name__ == "__main__":
    import os
    os.makedirs("cot_results", exist_ok=True)

    # Run the baseline for restaurant test set
    print("Running baseline on restaurant test set.")
    baseline_restaurant = batch_evaluate(combined_df, mode="direct_zero_shot")
    evaluate(baseline_restaurant)
    baseline_restaurant.to_csv("cot_results/results_baseline_restaurant.csv", index=False)

    # Run the baseline for laptop test set
    print("Running baseline on laptop test set.")
    baseline_laptop = batch_evaluate(laptop_df, mode="direct_zero_shot")
    evaluate(baseline_laptop)
    baseline_laptop.to_csv("cot_results/results_baseline_laptop.csv", index=False)

    # Run the CoT for restaurant test set
    print("Running CoT on restaurant test set.")
    cot_restaurant = batch_evaluate(combined_df, mode="few_shot_cot")
    evaluate(cot_restaurant)
    cot_restaurant.to_csv("cot_results/results_cot_restaurant.csv", index=False)

    # Run the CoT for laptop test set
    print("Running CoT on laptop test set.")
    cot_laptop = batch_evaluate(laptop_df, mode="few_shot_cot")
    evaluate(cot_laptop)
    cot_laptop.to_csv("cot_results/results_cot_laptop.csv", index=False)
