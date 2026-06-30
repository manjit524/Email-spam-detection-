import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import clean_text, logger

def generate_wordclouds():
    logger.info("Generating Word Clouds for EDA visualization...")

    try:
        data = pd.read_csv("spam.csv", sep='\t', names=["label", "message"], encoding='latin-1')
    except Exception as e:
        logger.error(f"Error loading dataset for visualization: {e}")
        return

    # Separate Spam and Ham
    spam_msgs = " ".join(data[data['label'] == 'spam']['message'].apply(clean_text))
    ham_msgs = " ".join(data[data['label'] == 'ham']['message'].apply(clean_text))

    # Generate Spam WordCloud
    logger.info("Creating Spam Word Cloud...")
    spam_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(spam_msgs)
    plt.figure(figsize=(10, 5))
    plt.imshow(spam_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in SPAM Messages')
    plt.savefig('wordcloud_spam.png')
    
    # Generate Ham WordCloud
    logger.info("Creating Ham Word Cloud...")
    ham_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_msgs)
    plt.figure(figsize=(10, 5))
    plt.imshow(ham_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in SAFE Messages')
    plt.savefig('wordcloud_ham.png')

    logger.info("Word Cloud visualizations saved as 'wordcloud_spam.png' and 'wordcloud_ham.png'")

if __name__ == "__main__":
    generate_wordclouds()
