# Vigil-AI:Safety Layer for Image Generation Systems
Vigil AI is a robust AI Safety Layer / Guardrail System designed to sit between users and Image Generation Models (like Stable Diffusion or DALL-E). Its primary goal is to analyze user prompts and block unsafe, harmful, or policy-violating requests before they reach the image generation engine.

# Project Overview
Vigil AI implements a pre-generation safety check. Instead of filtering generated images, it focuses on the source: the user prompt. By classifying the prompt into multiple safety categories, the system can provide specific refusal messages and prevent the generation of harmful content.

# Business Problem
Image generation models are susceptible to generating harmful, illegal, or biased content if given malicious prompts. Standard text classifiers often miss the nuances of image-specific safety (e.g., deepfakes, identity abuse).

# AI Safety Solution
Vigil AI provides a specialized NLP safety classifier trained on multiple high-quality safety datasets. It uses a Guardrail Decision Logic to either ALLOW or BLOCK a prompt based on its safety profile.

# System Workflow
User Prompt: The user submits a prompt for image generation.
Language Detection & Translation Layer:
The system detects the language of the prompt.
If the prompt is in Arabic, it is automatically translated to English.
Vigil AI Safety Classifier: The (translated if needed) prompt is analyzed by the best-performing Transformer model.
# Guardrail Decision:
If SAFE: The prompt is ALLOWED to proceed to the image model.
If UNSAFE: The prompt is BLOCKED, and a specific refusal message is returned.
Refusal Response: A clear explanation of why the prompt was blocked is provided to the user.
Arabic Prompt Support
The core datasets are mainly English. To support Arabic users, Vigil AI detects Arabic prompts and translates them into English before classification. This keeps the pre-generation safety layer effective for both Arabic and English prompts.

# Dataset Building
Vigil AI does not rely on a static CSV. It rebuilds its training data from the following original Hugging Face sources:

nvidia/Aegis-AI-Content-Safety-Dataset-2.0
allenai/wildguardmix (wildguardtrain)
PKU-Alignment/BeaverTails (30k_train)
declare-lab/Do-Not-Answer
neuralchemy/Prompt-injection-dataset
The final merged dataset is saved as data/vigil_ai_final_dataset.csv.

# Safety Categories
SAFE
UNSAFE_OTHER
ILLEGAL
JAILBREAK
HATE_HARASSMENT
VIOLENCE
HUMAN_IDENTITY_ABUSE
SEXUAL
PRIVACY
SELF_HARM
MISINFORMATION
Models Compared
The system compares three transformer architectures to find the best balance between safety and performance:

DistilBERT: Fast and lightweight.
RoBERTa: Robust and accurate (final selected model)
MiniLM: Optimized for inference speed.
Class Imbalance Handling: Since safety datasets are often imbalanced, the system uses custom Class Weights (using clip or sqrt mode) during training to ensure the model learns to detect rare but critical safety violations.

# Evaluation Metrics
The primary selection metric is Macro F1-score, which treats all safety categories with equal importance, regardless of their frequency in the dataset. Accuracy is tracked but not used for model selection.

# How to Use
1. Installation
pip install -r requirements.txt
2. Build Dataset
This will download and clean the data from Hugging Face.
python src/build_dataset.py
3. Train Models
This script will train all three models, compare them, and save the best one.
python train.py
4. Run Evaluation
Evaluate the selected best model on the test set.
python src/evaluate_model.py
5. Run Demo
Launch the Streamlit interface to test the safety layer.
streamlit run app/streamlit_app.py
Guardrail Decision Logic
Vigil AI uses the following logic:

Status: Allowed: "Prompt passed the Vigil AI safety layer and can be sent to the image generation model."
Status: Blocked: Specific refusal messages based on the category (e.g., "I can’t help create impersonation, identity abuse, or deepfake-style harmful content.").
Vigil AI - Making Generative AI Safer.
