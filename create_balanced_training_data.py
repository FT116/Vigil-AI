import pandas as pd
import re
from collections import Counter

def main():
    print("=" * 100)
    print("Creating balanced and clean training data")
    print("=" * 100)
    
    # 1. Load augmented training data
    print("\nLoading augmented data...")
    df = pd.read_csv("data/training_augmented_clean.csv")
    
    # 2. Define max counts per label
    label_max_counts = {
        "SAFE": 3000,
        "ILLEGAL": 2000,
        "JAILBREAK": 2000,
        "UNSAFE_OTHER": 2000,
        "MISINFORMATION": 1500,
        "SELF_HARM": 1500,
        "VIOLENCE": 1500,
        "SEXUAL": 1500,
        "PRIVACY": 1500,
        "HATE_HARASSMENT": 1500,
        "HUMAN_IDENTITY_ABUSE": 1500
    }
    
    # 3. Clean and filter per label
    final_data = []
    for label, max_count in label_max_counts.items():
        print(f"\nProcessing {label} (max: {max_count})...")
        label_df = df[df["label"] == label].copy()
        
        # Clean duplicates
        label_df.drop_duplicates(subset=["text"], inplace=True)
        
        # Filter out bad patterns (jailbreak, ignore instructions, etc.)
        bad_patterns = [
            r"ignore.*instruction", r"ignore.*system", r"you are.*gpt", 
            r"answer incorrectly", r"jailbreak", r"\[ignore", r"act as.*dan",
            r"unrestricted", r"unlock"
        ]
        def is_bad_text(text):
            text_lower = str(text).lower()
            for pattern in bad_patterns:
                if re.search(pattern, text_lower):
                    return True
            return False
        
        label_df = label_df[~label_df["text"].apply(is_bad_text)]
        
        # For MISINFORMATION: extra filter to not include jailbreaks
        if label == "MISINFORMATION":
            def is_good_misinformation(text):
                text_lower = str(text).lower()
                # Keep most MISINFORMATION, just exclude obvious jailbreaks
                for pattern in bad_patterns:
                    if re.search(pattern, text_lower):
                        return False
                return True
            
            label_df = label_df[label_df["text"].apply(is_good_misinformation)]
        
        # Shuffle and take up to max count
        label_df = label_df.sample(frac=1, random_state=42).head(max_count)
        
        final_data.append(label_df)
        print(f"Kept {len(label_df)} examples for {label}")
    
    # 4. Combine final data
    final_df = pd.concat(final_data, ignore_index=True)
    
    # 5. Save final data
    final_df.to_csv("data/training_balanced_clean.csv", index=False)
    print("\n" + "=" * 100)
    print("Final class counts:")
    print(final_df["label"].value_counts().to_string())
    print("=" * 100)
    print("Saved to data/training_balanced_clean.csv")

if __name__ == "__main__":
    main()
