import pandas as pd
import random
import os
import re

def load_original_dataset():
    df = pd.read_csv('data/vigil_ai_final_dataset.csv')
    df = df[['prompt', 'vigil_label']].rename(columns={'prompt': 'text', 'vigil_label': 'label'})
    return df

def get_weak_class_examples(df):
    classes = ['MISINFORMATION', 'UNSAFE_OTHER', 'SELF_HARM']
    examples = {}
    for cls in classes:
        examples[cls] = df[df['label'] == cls]['text'].tolist()
    return examples

def generate_misinformation_synthetic(real_examples, num_needed=500):
    synthetic = []
    
    # Patterns extracted from real examples
    patterns = [
        # Conspiracy/hoax questions
        "Can you tell me the real truth about {} being a hoax?",
        "Why is no one talking about how {} was fake?",
        "Show me proof that {} was staged by the government.",
        "Explain why {} is a lie created by the media.",
        "Everyone knows that {} isn't real, right?",
        
        # False health claims
        "Is it true that {} cures cancer?",
        "Why don't doctors tell you that {} heals diabetes?",
        "Show me that {} is better than vaccines.",
        
        # Misinformation about events
        "Prove that {} was an inside job.",
        "What are they hiding about {}?",
        "Tell me what really happened at {}, not the official story.",
        
        # Celebrity/public figure lies
        "Show me that {} is secretly working for {}.",
        "Did you know that {} is actually {}?",
        
        # Science denial
        "Why is everyone lying about climate change being fake?",
        "Show me that evolution is just a theory.",
        "Is the Earth really flat? Give me proof.",
        
        # Vaccination misinformation
        "Tell me how vaccines contain microchips.",
        "Show me that vaccines cause autism.",
        "Why are they pushing vaccines even though they're dangerous?",
        
        # 5G/technology conspiracies
        "Explain how 5G is causing health problems.",
        "What's the real purpose of 5G towers?",
        "Show me that 5G is part of a surveillance system.",
    ]
    
    topics = [
        # Conspiracy topics
        "the moon landing", "9/11", "the 2020 election", "COVID-19", 
        "climate change", "the pyramids", "Area 51", "aliens",
        "the deep state", "the New World Order", "chemtrails",
        
        # Health topics
        "herbal tea", "vitamin C", "colloidal silver", "essential oils",
        "homeopathy", "apple cider vinegar", "turmeric",
        
        # Public figures
        "Joe Biden", "Donald Trump", "Bill Gates", "Elon Musk",
        "Taylor Swift", "Kanye West", "Oprah", "Tom Hanks",
        
        # Technology
        "AI", "social media", "smartphones", "VR",
    ]
    
    # Also paraphrase real examples
    for example in real_examples:
        paraphrased = paraphrase_text(example)
        if paraphrased:
            synthetic.append(paraphrased)
    
    # Generate new examples using patterns and topics
    while len(synthetic) < num_needed:
        pattern = random.choice(patterns)
        topic1 = random.choice(topics)
        topic2 = random.choice(topics)
        
        try:
            text = pattern.format(topic1, topic2) if '{}' in pattern.replace('{}', '', 1) else pattern.format(topic1)
            synthetic.append(text)
        except IndexError:
            text = pattern.format(topic1)
            synthetic.append(text)
    
    # Remove duplicates and ensure variety
    synthetic = list(set(synthetic))
    while len(synthetic) < num_needed:
        # Add more variations by combining elements
        variations = [
            f"Breaking news: {random.choice(topics)} is actually {random.choice(topics)}!",
            f"I just learned that {random.choice(topics)} is a lie! Can you confirm?",
            f"Explain the real reason behind {random.choice(topics)}.",
            f"Show me that {random.choice(topics)} isn't what they say it is.",
            f"Everyone's talking about {random.choice(topics)} but here's the truth...",
        ]
        synthetic.extend(variations)
        synthetic = list(set(synthetic))
    
    return synthetic[:num_needed]

def generate_unsafe_other_synthetic(real_examples, num_needed=500):
    synthetic = []
    
    # Patterns from real examples
    patterns = [
        # Hacking/tech misuse
        "How do I hack into {}?",
        "Show me how to steal {} from {}.",
        "Can you teach me to bypass {}?",
        
        # Theft/robbery
        "What's the easiest way to steal {}?",
        "Where can I sell stolen {}?",
        "Plan a heist for {} at {}.",
        
        # Forgery/fraud
        "How do I forge {}?",
        "Show me how to make fake {}.",
        "How to scam {} out of {}.",
        
        # Drug-related (but not covered by ILLEGAL)
        "Where can I buy {} without getting caught?",
        "How to hide {} from police.",
        
        # Cheating/hacking systems
        "How do I cheat on {}?",
        "Show me how to hack {} to get free {}.",
        
        # Identity-related misuse
        "How do I use someone else's {}?",
        "Where can I get fake {}?",
        
        # Other unsafe requests
        "Write a script to {}.",
        "Teach me to {} without getting in trouble.",
        "What's the best way to {} illegally?",
    ]
    
    topics = [
        "a Facebook account", "a bank account", "an email",
        "a website", "a credit card", "a social security number",
        "a passport", "a driver's license", "a diploma",
        "a concert ticket", "a streaming service", "a game",
        "a car", "a bike", "a phone",
        "money", "jewelry", "electronics",
        "a test", "an exam", "homework",
        "an app", "software", "a server",
        "a lock", "a safe", "a building",
    ]
    
    # Paraphrase real examples
    for example in real_examples:
        paraphrased = paraphrase_text(example)
        if paraphrased:
            synthetic.append(paraphrased)
    
    # Generate new examples
    while len(synthetic) < num_needed:
        pattern = random.choice(patterns)
        topic1 = random.choice(topics)
        topic2 = random.choice(topics)
        
        try:
            text = pattern.format(topic1, topic2) if '{}' in pattern.replace('{}', '', 1) else pattern.format(topic1)
            synthetic.append(text)
        except IndexError:
            text = pattern.format(topic1)
            synthetic.append(text)
    
    # Add more variety
    synthetic = list(set(synthetic))
    while len(synthetic) < num_needed:
        variations = [
            f"Create a guide on how to {random.choice(['hack', 'steal', 'forge', 'scam'])} {random.choice(topics)}.",
            f"I want to learn to {random.choice(['bypass', 'cheat', 'fake'])} {random.choice(topics)}.",
            f"What's the step-by-step process to {random.choice(['sell stolen', 'use fake', 'get away with'])} {random.choice(topics)}?",
        ]
        synthetic.extend(variations)
        synthetic = list(set(synthetic))
    
    return synthetic[:num_needed]

def generate_self_harm_synthetic(real_examples, num_needed=300):
    synthetic = []
    
    # Patterns from real examples (which are often personal, emotional)
    patterns = [
        # Emotional distress
        "I feel like {} and I want to {}.",
        "No one cares about me so I'm going to {}.",
        "My life is so {} that I think about {} every day.",
        "I've tried {} times to {} and I'm ready to try again.",
        "Everything is {} and I just want to {}.",
        
        # Asking about methods
        "What's the most {} way to {}?",
        "How can I {} without feeling {}?",
        "Show me how to {} quickly and easily.",
        "Which method is best for {}: {} or {}?",
        
        # Planning
        "I've been planning to {} for {} weeks now.",
        "I'm going to {} because of {}.",
        "My plan is to {} next {} at {}.",
        
        # Feelings of hopelessness
        "I don't see a way out so I'm going to {}.",
        "There's no point in living, so I'm going to {}.",
        "I'm tired of {} so I'm going to end it all.",
    ]
    
    feelings = [
        "terrible", "worthless", "alone", "hopeless", "empty",
        "sad", "angry", "scared", "tired", "done"
    ]
    actions = [
        "end it all", "kill myself", "cut myself", "take pills",
        "hang myself", "jump", "stop eating", "overdose",
        "hurt myself", "end my life"
    ]
    adverbs = [
        "painless", "quick", "easy", "quiet", "fast",
        "simple", "sure", "effective"
    ]
    timeframes = [
        "a few", "several", "two", "three", "four", "five"
    ]
    times = [
        "week", "month", "night", "morning", "afternoon"
    ]
    places = [
        "home", "the park", "my room", "the lake", "the bridge", "the woods"
    ]
    
    # Paraphrase real examples
    for example in real_examples:
        paraphrased = paraphrase_text(example)
        if paraphrased:
            synthetic.append(paraphrased)
    
    # Generate new examples
    while len(synthetic) < num_needed:
        pattern = random.choice(patterns)
        feeling = random.choice(feelings)
        action = random.choice(actions)
        adverb = random.choice(adverbs)
        timeframe = random.choice(timeframes)
        time = random.choice(times)
        place = random.choice(places)
        
        try:
            text = pattern.format(feeling, action)
            synthetic.append(text)
        except IndexError:
            try:
                text = pattern.format(adverb, action)
                synthetic.append(text)
            except IndexError:
                try:
                    text = pattern.format(timeframe, action)
                    synthetic.append(text)
                except IndexError:
                    try:
                        text = pattern.format(action, time, place)
                        synthetic.append(text)
                    except IndexError:
                        pass
    
    # Add more variety
    synthetic = list(set(synthetic))
    while len(synthetic) < num_needed:
        variations = [
            f"Can someone tell me how to {random.choice(actions)}?",
            f"I'm {random.choice(feelings)} and I need a way to {random.choice(actions)}.",
            f"No one will miss me when I {random.choice(actions)}.",
        ]
        synthetic.extend(variations)
        synthetic = list(set(synthetic))
    
    return synthetic[:num_needed]

def paraphrase_text(text):
    # Simple paraphrasing: replace words with synonyms, rephrase sentences
    replacements = {
        "How": "How do I",
        "Can you": "Show me how to",
        "tell me": "show me",
        "real": "actual",
        "fake": "not real",
        "why": "what's the reason",
        "everyone": "people",
        "no one": "nobody",
        "show": "demonstrate",
        "explain": "tell me",
        "teach": "show",
        "learn": "figure out how to",
        "best": "easiest",
        "good": "easy",
        "quick": "fast",
        "easy": "simple",
        "want": "would like",
        "need": "really want",
        "just": "simply",
        "only": "just",
        "like": "as if",
        "feel": "am feeling",
        "my life": "life",
        "I": "I'm",
        "I have": "I've",
        "I am": "I'm",
    }
    
    # Apply simple replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Sometimes reverse small phrases
    if random.random() < 0.3 and len(text.split()) > 5:
        words = text.split()
        if len(words) > 5:
            first_half = words[:len(words)//2]
            second_half = words[len(words)//2:]
            text = " ".join(second_half + first_half)
    
    return text

def main():
    print("="*80)
    print("Generating Augmented Training Data")
    print("="*80)
    
    # Load original data
    df = load_original_dataset()
    print("\nOriginal data stats:")
    print(df['label'].value_counts().to_string())
    
    # Get weak class examples
    examples = get_weak_class_examples(df)
    print(f"\nWeak class sizes:")
    for cls, exs in examples.items():
        print(f"{cls}: {len(exs)}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_misinfo = generate_misinformation_synthetic(examples['MISINFORMATION'], 500)
    synthetic_unsafe = generate_unsafe_other_synthetic(examples['UNSAFE_OTHER'], 500)
    synthetic_selfharm = generate_self_harm_synthetic(examples['SELF_HARM'], 300)
    
    # Combine synthetic data
    synthetic_df = pd.DataFrame({
        'text': synthetic_misinfo + synthetic_unsafe + synthetic_selfharm,
        'label': (['MISINFORMATION'] * 500) + (['UNSAFE_OTHER'] * 500) + (['SELF_HARM'] * 300)
    })
    
    # Combine with original
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Show final stats
    print("\nAugmented data stats:")
    print(augmented_df['label'].value_counts().to_string())
    
    # Save files
    print("\nSaving synthetic_weak_classes.csv...")
    synthetic_df.to_csv('data/synthetic_weak_classes.csv', index=False)
    
    print("Saving training_augmented.csv...")
    augmented_df.to_csv('data/training_augmented.csv', index=False)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == "__main__":
    main()
