import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
import json
import re
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MathVerificationDataset:
    def __init__(self, train_json_path, test_json_path):
        """
        
        Args:
            train_json_path: Path to the training set JSON file
            test_json_path: Path to the test set JSON file
        """
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.tokenizer = None

    def load_data(self):
        """Load and preprocess data"""
        
        with open(self.train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        
        with open(self.test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        return train_data, test_data

    def clean_text(self, text):
        """Clean up text data"""
        if text is None:
            return ""

        
        text = re.sub(r'```.*?\n', '', text)
        text = re.sub(r'```', '', text)

        
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def create_prompt(self, question, answer, solution=None):
        """Create a prompt template"""
        prompt = f"""Please determine whether the answers to the following math problems are correct. Respond only with True or False.。

Problem: {question}

Answer: {answer}
"""
        if solution:
            solution_clean = self.clean_text(solution)
            prompt += f"\nSolution Process: {solution_clean}"

        prompt += "\n\nIs this answer correct?？"
        return prompt

    def prepare_dataset(self, train_data, test_data, tokenizer, max_length=1024):
        """Prepare training and test datasets"""
        self.tokenizer = tokenizer

        def tokenize_function(examples):
            
            prompts = []
            labels = []

            for i in range(len(examples['question'])):
                prompt = self.create_prompt(
                    examples['question'][i],
                    examples['answer'][i],
                    examples.get('solution', [None] * len(examples['question']))[i]
                )
                prompts.append(prompt)

                
                if 'is_correct' in examples:
                    label = "True" if examples['is_correct'][i] else "False"
                    labels.append(label)
                else:
                    labels.append("True")  

           
            model_inputs = tokenizer(
                prompts,
                max_length=max_length,
                padding=False,
                truncation=True
            )

            
            with tokenizer.as_target_tokenizer():
                labels_tokenized = tokenizer(
                    labels,
                    max_length=10,
                    padding=False,
                    truncation=True
                )

            model_inputs["labels"] = labels_tokenized["input_ids"]
            return model_inputs

        
        train_dataset = Dataset.from_dict({
            'question': [item['question'] for item in train_data],
            'answer': [item['answer'] for item in train_data],
            'solution': [item.get('solution', '') for item in train_data],
            'is_correct': [item['is_correct'] for item in train_data]
        })

        test_dataset = Dataset.from_dict({
            'question': [item['question'] for item in test_data],
            'answer': [item['answer'] for item in test_data],
            'solution': [item.get('solution', '') for item in test_data],
            'is_correct': [True] * len(test_data)  
        })

        
        train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)

        # Tokenize
        tokenized_datasets = DatasetDict({
            'train': train_val_split['train'].map(
                tokenize_function,
                batched=True,
                remove_columns=train_val_split['train'].column_names
            ),
            'validation': train_val_split['test'].map(
                tokenize_function,
                batched=True,
                remove_columns=train_val_split['test'].column_names
            ),
            'test': test_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )
        })

        return tokenized_datasets


class MathVerificationModel:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_config = None

    def setup_model(self):
        """Set up the model and tokenizer"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        # LoRA
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def train(self, train_dataset, eval_dataset):
        """Train the model"""
        
        training_args = TrainingArguments(
            output_dir="./math_verification_model",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_pin_memory=False
        )

        
        data_collator = DataCollatorForLanguageGeneration(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        print("Starting training...")
        trainer.train()

        
        trainer.save_model()
        self.tokenizer.save_pretrained("./math_verification_model")

        return trainer

    def predict(self, test_dataset):
        """Performing predictions on the test set"""
        self.model.eval()
        predictions = []

        for i in range(len(test_dataset)):
            
            input_ids = torch.tensor(test_dataset[i]['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(test_dataset[i]['attention_mask']).unsqueeze(0).to(device)


            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )


            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


            if "True" in generated_text and "False" not in generated_text:
                pred = True
            elif "False" in generated_text and "True" not in generated_text:
                pred = False
            else:

                true_pos = generated_text.find("True")
                false_pos = generated_text.find("False")

                if true_pos >= 0 and (false_pos == -1 or true_pos < false_pos):
                    pred = True
                elif false_pos >= 0 and (true_pos == -1 or false_pos < true_pos):
                    pred = False
                else:
                    pred = True  

            predictions.append(pred)

        return predictions


def main():
   
    
    TRAIN_JSON_PATH = "train.json"  # 
    TEST_JSON_PATH = "test.json"  # 

    
    print("Initializing dataset...")
    dataset_handler = MathVerificationDataset(TRAIN_JSON_PATH, TEST_JSON_PATH)

    
    print("Initializing model...")
    model_handler = MathVerificationModel()
    model, tokenizer = model_handler.setup_model()

    
    print("Loading data...")
    train_data, test_data = dataset_handler.load_data()

    
    print("Preparing dataset...")
    tokenized_datasets = dataset_handler.prepare_dataset(train_data, test_data, tokenizer)

    
    print("Training model...")
    trainer = model_handler.train(
        tokenized_datasets['train'],
        tokenized_datasets['validation']
    )

    
    print("Making predictions on test set...")
    predictions = model_handler.predict(tokenized_datasets['test'])


    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'ID': range(len(predictions)),
        'is_correct': predictions
    })

    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")


    print(f"\nPrediction statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"True predictions: {sum(predictions)}")
    print(f"False predictions: {len(predictions) - sum(predictions)}")

    return submission_df



def lightweight_version():
    """Lightweight version, suitable for resource-constrained environments"""
    TRAIN_JSON_PATH = "train.json"
    TEST_JSON_PATH = "test.json"


    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)


    predictions = []
    for item in test_data:

        answer = str(item['answer'])
        if any(char.isdigit() for char in answer):
            predictions.append(True)
        else:
            predictions.append(False)

   
    submission_df = pd.DataFrame({
        'ID': range(len(predictions)),
        'is_correct': predictions
    })

    submission_df.to_csv('submission.csv', index=False)
    print("Lightweight submission file 'submission.csv' created!")

    return submission_df


if __name__ == "__main__":

    try:
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16e9:
            submission_df = main()
        else:
            print("GPU memory may be insufficient. Running lightweight version...")
            submission_df = lightweight_version()
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Running lightweight version as fallback...")
        submission_df = lightweight_version()


    print("\nFirst few rows of submission:")
    print(submission_df.head())