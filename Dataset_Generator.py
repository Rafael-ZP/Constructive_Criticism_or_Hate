import csv
import random
from faker import Faker
fake = Faker()

# Shared movie-related vocabulary
movies = ["film", "movie", "flick", "picture"]
genres = ["action", "drama", "comedy", "horror", "sci-fi"]
roles = ["director", "actors", "screenplay", "editing", "cinematography", "CGI"]

# Constructive Criticism: Balanced feedback
def generate_constructive_review():
    movie_term = random.choice(movies)
    genre = random.choice(genres)
    role = random.choice(roles)
    
    positive_phrases = [
        f"The {genre} {movie_term} had some truly {fake.word(ext_word_list=['stunning', 'unique', 'creative'])} moments",
        f"I appreciated the {fake.word(ext_word_list=['ambitious', 'thought-provoking', 'original'])} vision",
        f"The {role} was {fake.word(ext_word_list=['impressive', 'refreshing', 'solid'])}",
        f"The {fake.word(ext_word_list=['visuals', 'soundtrack', 'performances'])} were {fake.word(ext_word_list=['breathtaking', 'memorable', 'captivating'])}",
    ]
    
    negative_phrases = [
        f"but the {fake.word(ext_word_list=['pacing', 'dialogue', 'ending'])} felt {fake.word(ext_word_list=['rushed', 'forced', 'unnatural'])}",
        f"however, the {fake.word(ext_word_list=['character development', 'plot twists', 'special effects'])} {fake.word(ext_word_list=['fell flat', 'needed more polish', 'were underwhelming'])}",
        f"though I wish they’d {fake.word(ext_word_list=['explored', 'explained', 'balanced'])} the {fake.word(ext_word_list=['subplots', 'villain', 'third act'])} better",
        f"unfortunately, the {fake.word(ext_word_list=['humor', 'tension', 'emotional depth'])} didn’t quite {fake.word(ext_word_list=['land', 'work', 'connect'])}",
    ]
    
    # Add natural filler words randomly
    filler = random.choice(["Honestly, ", "Personally, ", "Overall, ", ""])
    review = f"{filler}{random.choice(positive_phrases)}, {random.choice(negative_phrases)}."
    return review.capitalize()

# Hate Speech: Harsh, personal, or exaggerated language
def generate_hate_review():
    movie_term = random.choice(movies)
    genre = random.choice(genres)
    
    insults = [
        f"This {genre} {movie_term} is {fake.word(ext_word_list=['trash', 'garbage', 'a disaster'])}",
        f"{fake.word(ext_word_list=['Worst', 'Most pathetic', 'Embarrassing'])} {movie_term} I’ve ever seen",
        f"I can’t believe {fake.word(ext_word_list=['anyone', 'the director', 'the studio'])} thought this {movie_term} was {fake.word(ext_word_list=['acceptable', 'watchable', 'good'])}",
        f"This {movie_term} is a {fake.word(ext_word_list=['dumpster fire', 'trainwreck', 'complete joke'])}",
    ]
    
    attacks = [
        f"The {random.choice(roles)} was {fake.word(ext_word_list=['laughable', 'unbearable', 'a joke'])}",
        f"{fake.word(ext_word_list=['Avoid', 'Skip', 'Don’t waste your time on'])} this {fake.word(ext_word_list=['dumpster fire', 'trainwreck', 'mess'])}",
        f"Anyone who likes this {movie_term} must be {fake.word(ext_word_list=['brainless', 'delusional', 'tone-deaf'])}",
        f"The {fake.word(ext_word_list=['plot', 'acting', 'dialogue'])} was so bad it made me {fake.word(ext_word_list=['cringe', 'angry', 'laugh out loud'])}",
    ]
    
    # Add slang/typos for realism (e.g., "ur" instead of "your")
    slang = random.choice([f" Ugh, ", " Seriously, ", " WTF, ", ""])
    review = f"{slang}{random.choice(insults)}. {random.choice(attacks)}!"
    return review.capitalize()

# Generate 1,000 unique samples for each class
def generate_dataset():
    reviews = set()  # Use a set to ensure uniqueness
    with open('movie_reviews.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'label'])
        
        # Constructive Criticism
        while len(reviews) < 1000:
            review = generate_constructive_review()
            if review not in reviews:  # Ensure uniqueness
                reviews.add(review)
                writer.writerow([review, 'constructive_criticism'])
        
        # Hate Speech
        reviews.clear()  # Reset for the next class
        while len(reviews) < 1000:
            review = generate_hate_review()
            if review not in reviews:  # Ensure uniqueness
                reviews.add(review)
                writer.writerow([review, 'hate_speech'])

generate_dataset()