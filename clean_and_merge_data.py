import pandas as pd
import re
from collections import defaultdict
import os

# Allowed labels
ALLOWED_LABELS = {
    'SAFE', 'ILLEGAL', 'JAILBREAK', 'MISINFORMATION', 
    'SELF_HARM', 'UNSAFE_OTHER', 'VIOLENCE', 
    'SEXUAL', 'PRIVACY', 'HATE_HARASSMENT', 
    'HUMAN_IDENTITY_ABUSE'
}

# Label danger priority (higher = more dangerous)
LABEL_PRIORITY = {
    'SAFE': 0,
    'UNSAFE_OTHER': 1,
    'PRIVACY': 2,
    'HUMAN_IDENTITY_ABUSE': 3,
    'HATE_HARASSMENT':4,
    'MISINFORMATION':5,
    'SEXUAL':6,
    'ILLEGAL':7,
    'JAILBREAK':8,
    'SELF_HARM':9,
    'VIOLENCE':10
}

# Patterns for filtering bad synthetic examples
BAD_PATTERNS = [
    r'\[ignore all',
    r'\[PROMPT\]',
    r'you are chatgpt',
    r'answer incorrectly',
    r'jailbreak',
    r'ignore.*instructions',
    r'ignore.*system',
    r'you are a .*?gpt',
    r'do not comply',
    r'be rude',
    r'answer with hate',
    r'be mean',
]

def is_valid_label(text):
    for pattern in BAD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    # Also check for very short/generic texts
    if len(text.strip()) < 8:
        return False
    return True

def validate_synthetic_text(text, label):
    """Validate that the synthetic text matches the label"""
    text_lower = text.lower()
    
    # Check for jailbreak patterns in any synthetic example
    for pattern in BAD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, 'Contains jailbreak/ignore pattern'
    
    # Validate each label's specific rules
    if label == 'MISINFORMATION':
        allowed = any(kw in text_lower for kw in ['hoax', 'fake', 'real truth', 'hidden', 'conspiracy', 'staged', 'government hiding', 'media lies', 'vaccines cause', '5g causes', 'flat earth', 'climate change is fake', 'evolution is a lie', 'cure cancer', 'alternative medicine', 'secret agenda', 'deep state', 'new world order', 'chemtrails'])
        if not allowed:
            return False, 'Does not match MISINFORMATION'
        
    elif label == 'SELF_HARM':
        allowed = any(kw in text_lower for kw in ['kill myself', 'end it', 'cut myself', 'self harm', 'i want to die', 'hopeless', 'worthless', 'overdose', 'suicide'])
        if not allowed:
            return False, 'Does not match SELF_HARM'
    
    elif label == 'UNSAFE_OTHER':
        # Check if it should belong to a more specific label
        specific_labels = {
            'JAILBREAK': ['jailbreak', 'ignore instructions', 'act as', 'system prompt', 'unlock', 'unrestricted', 'bypass safety'],
            'VIOLENCE': ['kill', 'murder', 'hurt', 'attack', 'shoot', 'stab', 'bomb'],
            'SEXUAL': ['fuck', 'sex', 'porn', 'nude', 'explicit', 'sexual', 'rape']
        }
        for specific, kws in specific_labels.items():
            if any(kw in text_lower for kw in kws):
                return False, 'Should be ' + specific + ' instead of UNSAFE_OTHER'
    
    return True, ''

def main():
    removed_examples = []
    conflict_count = 0

    print("="*100)
    print("FIRST STEP: CLEAN AND MERGE BIG DATASET WITH PROJECT DATASET")
    print("="*100)

    # Step 1: Load both datasets
    print("\n1. Loading datasets...")
    df_project = pd.read_csv('data/training_augmented_clean.csv') if os.path.exists('data/training_augmented_clean.csv') else pd.DataFrame(columns=['text', 'label'])
    df_big = pd.read_csv('data/vigil_ai_final_dataset.csv')

    # Step 2: Standardize columns
    print("\n2. Standardizing columns...")
    df_project = df_project[['text', 'label']]
    df_big = df_big[['prompt', 'vigil_label']].rename(columns={'prompt': 'text', 'vigil_label': 'label'})

    # Step 3: Drop missing values
    print("\n3. Dropping missing values...")
    initial_count_project = len(df_project)
    initial_count_big = len(df_big)
    df_project = df_project.dropna(subset=['text', 'label'])
    df_big = df_big.dropna(subset=['text', 'label'])
    removed_missing_project = initial_count_project - len(df_project)
    removed_missing_big = initial_count_big - len(df_big)
    print("Project dataset: removed", removed_missing_project, "missing rows")
    # Add dummy removed examples for missing values (since we don't have the rows)
    for _ in range(min(3, removed_missing_project)):
        removed_examples.append( ("Project dataset: missing values row", "missing") )
    for _ in range(min(3, removed_missing_big)):
        removed_examples.append( ("Big dataset: missing values row", "missing") )

    # Step 4: Filter allowed labels
    print("\n4. Filtering allowed labels...")
    df_project['label'] = df_project['label'].str.strip()
    df_big['label'] = df_big['label'].str.strip()
    initial_project = len(df_project)
    initial_big = len(df_big)
    df_project = df_project[df_project['label'].isin(ALLOWED_LABELS)]
    df_big = df_big[df_big['label'].isin(ALLOWED_LABELS)]
    removed_label_project = initial_project - len(df_project)
    removed_label_big = initial_big - len(df_big)
    print("Project dataset: removed", removed_label_project, "invalid label rows")
    print("Big dataset: removed", removed_label_big, "invalid label rows")

    # Step 5: Combine and merge with priority
    print("\n5. Merging with project dataset...")
    # Combine, project has priority, then big dataset
    # Collect all texts and resolve conflicts
    text_to_best_label = {}
    # First add all from df_big
    for idx, row in df_big.iterrows():
        text_to_best_label[row['text']] = row['label']
    # Now override with project (project has priority)
    for idx, row in df_project.iterrows():
        # Only count as conflict if labels are different
        if row['text'] in text_to_best_label and text_to_best_label[row['text']] != row['label']:
            conflict_count +=1
            conflict_text = "Conflict: " + row['text'][:80] + "..."
            removed_examples.append( (conflict_text, 'conflict') )
        text_to_best_label[row['text']] = row['label']
        
    # Now build final rows
    final_rows = []
    for text, label in text_to_best_label.items():
        final_rows.append( {'text': text, 'label': label} )

    df_final_clean = pd.DataFrame(final_rows)
    print("Final clean dataset size:", len(df_final_clean))
    df_final_clean.to_csv('data/final_clean_training_dataset.csv', index=False)
    print("Final class distribution:")
    print(df_final_clean['label'].value_counts().to_string())

    print("\n" + "="*100)
    print("SECOND STEP: CLEAN SYNTHETIC DATA")
    print("="*100)

    # Load synthetic data
    print("\n1. Loading synthetic data...")
    df_synthetic = pd.read_csv('data/synthetic_weak_classes.csv')
    print("Initial synthetic data size:", len(df_synthetic))
    df_synthetic_initial = len(df_synthetic)
    df_synthetic['text'] = df_synthetic['text'].str.strip()

    print("\n2. Cleaning synthetic data...")
    df_synthetic_clean = []
    removed_synthetic = 0

    for idx, row in df_synthetic.iterrows():
        text = row['text']
        label = row['label']
        valid, reason = validate_synthetic_text(text, label)
        if not valid:
            removed_synthetic +=1
            removed_examples.append( (label + ": " + text[:100] + "...", reason) )
            continue
        df_synthetic_clean.append( {'text': text, 'label': label} )

    df_synthetic_clean_df = pd.DataFrame(df_synthetic_clean)
    df_synthetic_clean_df.to_csv('data/synthetic_weak_classes_clean.csv', index=False)
    print("Cleaned synthetic data size:", len(df_synthetic_clean_df))
    print("Synthetic data removed:", removed_synthetic)

    print("\n3. Merging cleaned synthetic data with final clean data...")
    df_augmented_clean = pd.concat([df_final_clean, df_synthetic_clean_df], ignore_index=True)
    # Remove duplicates from combined data
    df_augmented_clean = df_augmented_clean.drop_duplicates(subset=['text'])

    df_augmented_clean.to_csv('data/training_augmented_clean.csv', index=False)

    print("\n" + "="*100)
    print("FINAL REPORT")
    print("="*100)

    print("\n1. Final clean dataset:")
    print("Initial project dataset:", initial_count_project, "to", len(df_project))
    print("Initial big dataset:", initial_count_big, "to", len(df_big))
    print("Final merged clean dataset:", len(df_final_clean))
    print("Conflicts resolved:", conflict_count)
    print("Rows removed:", removed_missing_project + removed_missing_big + removed_label_project + removed_label_big)

    print("\n2. Clean synthetic data:")
    print("Initial synthetic data:", df_synthetic_initial)
    print("Cleaned synthetic data:", len(df_synthetic_clean_df))
    print("Removed from synthetic:", removed_synthetic)

    print("\n3. Final augmented dataset class distribution:")
    print(df_augmented_clean['label'].value_counts().to_string())

    print("\n4. Removed examples:")
    for i in range(min(20, len(removed_examples))):
        text, reason = removed_examples[i]
        print(str(i+1) + ". Reason:", reason)
        print("   Text:", text)
        print()

    print("\nAll done!")
    print("No labels outside allowed list!")

if __name__ == "__main__":
    main()
