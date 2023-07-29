import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
import torch

from dotenv import load_dotenv
from nltk.sentiment import SentimentIntensityAnalyzer

from typing import Tuple, Union

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found.")
openai.api_key = openai_api_key


def get_env_variable(name, default=None):
    """
    Returns the value of the environment variable with the given name.
    First tries to fetch it from Streamlit secrets, and if not available,
    falls back to the local environment. If it's not found in either place,
    returns the default value if provided.
    
    Args:
        name (str): The name of the environment variable.
        default (str, optional): The default value to be returned in case the environment variable is not found.

    Returns:
        str: The value of the environment variable, or the default value.
    """
    if st.secrets is not None and name in st.secrets:
        # Fetch the secret from Streamlit secrets
        return st.secrets[name]
    else:
        # Try to get the secret from the local environment
        value = os.getenv(name)

        if value is None and default is not None:
            # If the environment variable is not found and a default value is provided,
            # return the default value
            return default
        else:
            return value

# Streamlit configurations
st.set_page_config(
    page_title="SentXMent - Sentiment Analysis",
    page_icon="🔬🗣️🚦",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/AdieLaine/',
        'Report a bug': 'https://github.com/AdieLaine/issues',
        'About': """
            ## SentXMent - Sentiment Analysis
            Welcome to SentXMent - Sentiment Analysis! This application is dedicated to analyzing the sentiment of various texts, including tweets, articles, and more.

            ## About Sentiment Analysis
            Sentiment analysis, also known as opinion mining, is the process of determining the sentiment or emotion expressed in a piece of text. It can help understand the overall sentiment, whether positive, negative, or neutral, of the given input text.

            ## How SentXMent Works
            SentXMent leverages the power of natural language processing (NLP) and machine learning to calculate sentiment scores for the provided text. It utilizes the NLTK SentimentIntensityAnalyzer to perform the initial sentiment analysis.

            Once the sentiment scores are obtained, SentXMent takes it a step further by interpreting the scores using the state-of-the-art OpenAI GPT-3.5 Turbo model. This AI-powered interpretation provides a detailed analysis of the sentiment, offering valuable insights into the text's emotional context.

            SentXMent also generates feedback or suggestions based on the sentiment analysis result. The feedback is crafted from a perspective influenced by analytical psychology and a logical mindset, providing an insightful interpretation of the sentiment in the text.

            ## Use Cases
            - Analyze the sentiment of tweets to gauge public opinion on a topic.
            - Evaluate the sentiment of customer reviews to understand product feedback.
            - Monitor sentiment in news articles to track public perception of current events.

            ## Your Contribution
            SentXMent is an open-source project, and we welcome contributions from developers and AI enthusiasts. If you'd like to contribute to SentXMent's development or explore the codebase, please visit our [GitHub](https://github.com/your-github-username/SentXMent) repository.

            We hope you find SentXMent useful for your sentiment analysis tasks. Happy analyzing!
        """
    }
)

def apply_css_and_display_title():
    """
    Apply the custom CSS style for the title and display the title. 

    The function creates a custom CSS style for the title, breaking it into three parts: 'Sent', '𝕏', and 'Ment', each with a unique color. 
    It then applies this style to the title and displays the title on the webpage using Streamlit's markdown functionality. 
    Following the title, a 'Sentiment Analysis' subtitle is displayed.
    """
    title_style = """
        <style>
            .title-text {
                text-align: center;
                margin-bottom: 0;
                padding: 10px;
                font-size: 59px;
                font-smoothing: antialiased;
                -webkit-font-smoothing: antialiased;
            }
            .letter-s {
                color: MediumSeaGreen;
            }
            .letter-x {
                color: Gainsboro;
                font-size: 69px;
                font-smoothing: antialiased;
                -webkit-font-smoothing: antialiased;
            }
            .letter-ment {
                color: Crimson;
            }
        </style>
    """
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown('<h1 class="title-text"><span class="letter-s">Sent</span><span class="letter-x">\U0001D54F</span><span class="letter-ment">Ment</span></h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center;">Sentiment Analysis for 𝕏</h3>', unsafe_allow_html=True)
    st.markdown("---")


def update_mean(
    current_mean: torch.Tensor,
    current_weight_sum: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update the mean and weight sum using the Welford formula.
    This function is inspired by the aggregation logic from the Twitter Algorithm.
    
    Args:
        current_mean (torch.Tensor): The current accumulated mean.
        current_weight_sum (torch.Tensor): The current weighted sum.
        value (torch.Tensor): The new value to be added to the mean.
        weight (torch.Tensor): The weight of the new value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated mean and weighted sum.
    """
    weight = torch.broadcast_to(weight, value.shape)
    current_weight_sum = current_weight_sum + torch.sum(weight)
    current_mean = current_mean + torch.sum((weight / current_weight_sum) * (value - current_mean))
    return current_mean, current_weight_sum


def update_stable_mean(
    mean_and_weight_sum: torch.Tensor, 
    value: torch.Tensor, 
    weight: Union[float, torch.Tensor] = 1.0
) -> None:
    """
    Update the stable mean using the 'update_mean' function.

    Args:
        mean_and_weight_sum (torch.Tensor): The tensor storing the current mean and weight sum.
        value (torch.Tensor): The new value to be added to the mean.
        weight (Union[float, torch.Tensor]): The weight of the new value, default is 1.0.

    Returns:
        None
    """
    mean, weight_sum = mean_and_weight_sum[0], mean_and_weight_sum[1]

    if not isinstance(weight, torch.Tensor):
        weight = torch.as_tensor(weight, dtype=value.dtype, device=value.device)

    mean_and_weight_sum[0], mean_and_weight_sum[1] = update_mean(
        mean, weight_sum, value, torch.as_tensor(weight)
    )

def calculate_stable_mean(values):
    """
    Calculate the stable mean of a list of values using the 'update_stable_mean' function.

    Args:
        values (list): The list of values.

    Returns:
        float: The stable mean of the values.
    """
    mean_and_weight_sum = torch.zeros(2)
    for value in values:
        update_stable_mean(mean_and_weight_sum, torch.tensor(value))
    return mean_and_weight_sum[0].item()


def analyze_sentiment(input_text: str, sia: SentimentIntensityAnalyzer = None) -> dict:
    """
    Analyzes the sentiment of the input text using NLTK's SentimentIntensityAnalyzer.

    This function calculates polarity scores that range between -1 and 1 for the input text, 
    where 1 signifies positive sentiment and -1 signifies negative sentiment. The scores are 
    based on a lexicon of words that have been preassigned scores that denote the sentiment 
    they carry. The function returns a dictionary that includes the compound score (an 
    aggregate sentiment score) as well as individual scores for the positive, negative, 
    and neutral sentiment of the text.

    Args:
        input_text (str): The input text to be analyzed.
        sia (SentimentIntensityAnalyzer, optional): An optional SentimentIntensityAnalyzer 
            object. If none is provided, a new one will be created.

    Returns:
        sentiment_scores (dict): A dictionary of the sentiment scores of the input text.
            The keys are 'neg', 'neu', 'pos', and 'compound', and the values are the corresponding scores.

    Raises:
        ValueError: If the input text is not a string or is empty.
    """
    # Check that input_text is a string and is not empty
    if not isinstance(input_text, str):
        raise ValueError("Input text must be a string")
    if input_text.strip() == "":
        raise ValueError("Input text must not be empty")

    # Create a SentimentIntensityAnalyzer object if none is provided
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    # Calculate sentiment scores
    sentiment_scores = sia.polarity_scores(input_text)

    return sentiment_scores


def detailed_feedback(sentiment_scores, prompt):
    """
    Generate feedback or suggestions based on the sentiment analysis result.

    This function takes the sentiment scores and the input text, sends them to the GPT-3 model as a 
    prompt, and asks the model to generate feedback or suggestions influenced by the analytical psychology 
    of Carl Jung and the logical mindset of Abraham Maslow, without directly mentioning their names.

    Args:
        sentiment_scores (dict): A dictionary of sentiment scores obtained from analyzing the input text.
            The keys are 'neg', 'neu', 'pos', and 'compound', and the values are the corresponding scores.
        prompt (str): The original text that was analyzed to obtain the sentiment scores.

    Returns:
        feedback (str): The feedback or suggestions generated by the GPT-3 model influenced by the perspectives 
        of Carl Jung and Abraham Maslow.
    """
    feedback_prompt = f"Given these sentiment scores '{sentiment_scores}' for the input text '{prompt}', provide feedback or suggestions in a manner that subtly embodies analytical psychology and the logical patterns that can be interpreted"

    feedback_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": feedback_prompt},
            {"role": "assistant", "content": "With a perspective of Carl Jung and Maslow's Hyierachy of Needs, provide feedback basded on these values that embrace complexity of human emotions and the desire for the right response, let's delve into these sentiment scores..."}
        ],
        temperature=0.7,
        max_tokens=1000,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    feedback = feedback_response['choices'][0]['message']['content']
    return feedback


def sentiment_x_interpreter(sentiment_scores, prompt):
    """
    Send sentiment scores to OpenAI's GPT-3 model for interpretation.

    This function constructs a prompt that includes the original text and its sentiment scores. 
    It then sends this prompt to the GPT-3 model, which generates a detailed analysis of the sentiment 
    in the text. The generated sentiment analysis result is returned as a string.

    Args:
        sentiment_scores (dict): A dictionary of sentiment scores obtained from analyzing the input text.
            The keys are 'neg', 'neu', 'pos', and 'compound', and the values are the corresponding scores.
        prompt (str): The original text that was analyzed to obtain the sentiment scores.

    Returns:
        sentiment_analysis_result (str): The interpreted sentiment analysis result generated by the GPT-3 model.
    """
    sentiment_interpretation_prompt = f"The sentiment scores for the input text '{prompt}' are '{sentiment_scores}'. Could you interpret these scores and provide a detailed analysis of the sentiment in the text?"

    sentiment_interpretation_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sentiment_interpretation_prompt},
            {"role": "assistant", "content": f"Based on the provided '{sentiment_scores}', I will reiterate the {prompt}. Then break down the {prompt} in an analysis and detail each part:"}
        ],
        temperature=0.7,
        max_tokens=1000,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    sentiment_analysis_result = sentiment_interpretation_response['choices'][0]['message']['content']
    return sentiment_analysis_result


def display_results(sentiment_score, stable_mean_sentiment_score, sentiment_analysis_result, feedback):
    """
    Display the sentiment score, the stable mean of sentiment scores, and the sentiment analysis result.

    This function displays the sentiment score and the stable mean of sentiment scores in two separate 
    columns on the web page. It then displays the sentiment analysis result as a chat message. The function 
    also displays an info box with a brief explanation of the methodology used to calculate the stable mean 
    of sentiment scores, and an expander with detailed information about the sentiment scores and the algorithms 
    used. Finally, it displays a toast message indicating the completion of the sentiment analysis and interpretation.

    Args:
        sentiment_score (float): The sentiment score to be displayed.
        stable_mean_sentiment_score (float): The stable mean of sentiment scores to be displayed.
        sentiment_analysis_result (str): The sentiment analysis result to be displayed.
        feedback (str): The feedback or suggestions to be displayed in an expander.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(help="The Sent𝕏Ment Sentiment Score represents the overall sentiment of the text input. It's a compound score calculated using NLTK's Sentiment Analysis, with values ranging from -1 (most negative sentiment) to +1 (most positive sentiment).",label="Sent𝕏Ment Sentiment Score", value=sentiment_score, delta=float(sentiment_score))
    with col2:
        st.metric(help="The Stable Mean Score, calculated using Welford's Method, represents a numerically stable mean of sentiment scores. This score provides a more reliable measure of overall sentiment, especially when dealing with large quantities of data or data with large variations.",label="Stable Mean Score (Welford's Method)", value=stable_mean_sentiment_score, delta=float(stable_mean_sentiment_score))

    st.markdown("---")

    with st.chat_message("assistant"):
        st.markdown(sentiment_analysis_result)

    with st.expander("Deeper Analysis and Feedback"):
        st.markdown(feedback)
        st.info("**Intended For Research/Educational Use** - Sent𝕏Ment uses logic to simulate and invoke a dynamic response. In this example we simulate the framework of analytical psychology and the principles of logical reasoning. The feedback and suggestions offered here echo the insights of a hybrid version of Carl Jung and Abraham Maslow.")
        


    #st.info("Sent𝕏Ment utilize the Welford algorithm and our own logic to calculate a stable mean of sentiment scores. This methodology, provides optimal and reliable results, we've included the aggregation logic from the Twitter Algorithm public repository.")

    with st.expander("Sentiment Scores and Algorithms"):
        st.markdown("""
        **Sent𝕏Ment Sentiment Score:**
        Sent𝕏Ment uses NLTK's Sentiment Analysis to evaluate the sentiment of a text input. The score is a compound value that signifies the overall sentiment of the text, normalized to range between -1 (most negative) and +1 (most positive).

        **Stable Mean Score (Welford's Method):**
        Sent𝕏Ment calculates a numerically stable mean using the Welford Algorithm. This method, robust to outliers and efficient for large datasets, yields more reliable results when dealing with large quantities of data or data with large variations. This logic is inspired by the aggregation method from the public Twitter Algorithm repository.

        **Sent𝕏Ment Logic:**
        The Sent𝕏Ment process involves performing sentiment analysis on a text input using NLTK, calculating the compound sentiment score, and then calculating the stable mean of this score using the implemented functions. This information is then interpreted using OpenAI's GPT-3 model, providing a detailed analysis of the sentiment in the text.

        **Welford's Algorithm:**
        Welford's Algorithm calculates a numerically stable mean, which is particularly useful in situations dealing with large amounts of data or data with large variations. The aggregation logic used here is inspired by methods used in industry settings, such as social media sentiment analysis, and has been adapted from the public Twitter Algorithm repository.
        """)
    st.toast("Sentiment analysis and interpretation complete.")


def main():
    """
    Main function to run the Sent𝕏Ment sentiment analysis application.

    This function orchestrates the entire sentiment analysis workflow, which includes the following steps:
    1. Apply CSS and display the title.
    2. Receive a text input from the user.
    3. Perform sentiment analysis on the input text using NLTK's SentimentIntensityAnalyzer.
    4. Calculate the stable mean of sentiment scores using the Welford algorithm.
    5. Send the sentiment scores to OpenAI's GPT-3 model for interpretation.
    6. Display the sentiment score, the stable mean of sentiment scores, and the sentiment analysis result.

    The function also displays toast messages to keep the user informed about the progress of the sentiment analysis.
    """
    apply_css_and_display_title()

    if prompt := st.chat_input("Paste a Tweet to analyze the sentiment..."):
        with st.spinner("Generating sentiment analysis result..."):
            st.toast("SentXMent is verifying sentiment values...")
            st.toast("Using NLTK sentiment data for this analysis...")

            sentiment_scores = analyze_sentiment(prompt.strip())
            sentiment_score = sentiment_scores['compound']
            stable_mean_sentiment_score = calculate_stable_mean([sentiment_score])
            sentiment_analysis_result = sentiment_x_interpreter(sentiment_scores, prompt)
            feedback = detailed_feedback(sentiment_scores, prompt)

            display_results(sentiment_score, stable_mean_sentiment_score, sentiment_analysis_result, feedback)

if __name__ == "__main__":
    main()
#pegn'g