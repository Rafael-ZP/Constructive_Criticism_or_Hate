# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np
import torch
import tiktoken
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import logging
logging.basicConfig(level=logging.DEBUG)

# Cell 2: Load dataset and preprocess labels
df = pd.read_csv("/Users/rafaelzieganpalg/Projects/SRP_Lab/Main_Proj/Dataset/movie_reviews_dataset.csv")
label_map = {"Constructive Criticism": 0, "Hate Speech": 1}
df["label"] = df["label"].map(label_map)

# Split dataset (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["review"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)

from transformers import AutoTokenizer

model_name = "microsoft/deberta-v3-base"  # Use the model's Hugging Face name

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Cell 4: Create PyTorch dataset class
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewsDataset(train_encodings, train_labels)
test_dataset = ReviewsDataset(test_encodings, test_labels)

# Cell 5: Load pre-trained DeBERTa model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./deberta_output",
    num_train_epochs=3,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=1,
    logging_strategy="steps",  # Log after every step
    log_level="info",  # Display logs
    disable_tqdm=False,  # Ensure progress bar shows
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"  # Prevent logging suppression
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate model on the test set
eval_results = trainer.evaluate(test_dataset)
print("Evaluation results:", eval_results)

# Cell 6: Generate predictions and visualize results
preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)

# Compute confusion matrix
cm = confusion_matrix(test_labels, preds)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix:\n", cm)
print("Normalized Confusion Matrix:\n", cm_normalized)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Constructive Criticism", "Hate Speech"],
            yticklabels=["Constructive Criticism", "Hate Speech"])
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("normalized_confusion_matrix.png")
plt.show()

# Extract training logs
logs = trainer.state.log_history
epochs = []
eval_acc = []
eval_f1 = []

# Store evaluation metrics for plotting
for log in logs:
    if "epoch" in log and "eval_accuracy" in log and "eval_f1" in log:
        epochs.append(log["epoch"])
        eval_acc.append(log["eval_accuracy"])
        eval_f1.append(log["eval_f1"])

# Plot accuracy per epoch
if epochs and eval_acc:
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, eval_acc, marker="o", label="Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_chart.png")
    plt.show()

# Plot F1 score per epoch
if epochs and eval_f1:
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, eval_f1, marker="o", color="orange", label="F1 Score")
    plt.title("Validation F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("f1_score_chart.png")
    plt.show()

# Save DeBERTa model
model.save_pretrained("deberta_model")
tokenizer.save_pretrained("deberta_model")
print("DeBERTa model and tokenizer saved successfully!")