# -*- coding: utf-8 -*-
import pandas as pd
import re
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from typing import Dict, List


class Llama3MathSFT:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _load_tokenizer(self):
        """Load Llama-3Tokenizer and configure"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" 
        self.tokenizer.add_eos_token = True
        return self.tokenizer

    def _load_model(self):
        """Loading the quantized Llama-3 8B model"""
        # 4-bit Quantization Configuration (Reduces VRAM Usage)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # GPU/CPU
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # LoRA
        lora_config = LoraConfig(
            r=8,  # LoRA
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Llama-3
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  
        return self.model

    def _preprocess_data(self, train_csv_path: str, test_csv_path: str):
        """Preprocess the training set and test set"""
        # Loading data
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        
        def clean_text(text: str) -> str:
            if pd.isna(text):
                return ""
            
            text = re.sub(r'<llm-code>.*?</llm-code>', '', text, flags=re.DOTALL)
            text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', text, flags=re.DOTALL)
            
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\.', '.', text)
            return text.strip()

        
        def format_train_sample(row):
            question = clean_text(row["question"])
            answer = clean_text(row["answer"])
            solution = clean_text(row["solution"])

            prompt = f"""Decide if the given answer to the math question is correct. Respond with True or False only.
Math Question: {question}
Given Answer: {answer}
Solution Explanation: {solution}
Is the answer correct? """

            
            label = "True" if row["is_correct"] else "False"
            return {"text": prompt + label}

        
        def format_test_sample(row, idx):
            question = clean_text(row["question"])
            answer = clean_text(row["answer"])
            solution = clean_text(row["solution"]) if "solution" in row.columns else ""

            prompt = f"""Decide if the given answer to the math question is correct. Respond with True or False only.
Math Question: {question}
Given Answer: {answer}
Solution Explanation: {solution}
Is the answer correct? """

            return {"text": prompt, "ID": idx}

        
        train_dataset = Dataset.from_pandas(train_df).map(format_train_sample)
        test_dataset = Dataset.from_pandas(test_df).enumerate().map(
            lambda x: format_test_sample(x["element"], x["index"])
        )

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        
        tokenized_train = tokenized_train.map(
            lambda x: {"labels": x["input_ids"][1:] + [self.tokenizer.pad_token_id]}
        )

        return tokenized_train, tokenized_test, test_df

    def train(self, train_csv_path: str, test_csv_path: str):
        """Training the SFT model"""
        
        self._load_tokenizer()
        self._load_model()

        
        tokenized_train, tokenized_test, _ = self._preprocess_data(train_csv_path, test_csv_path)

        
        training_args = TrainingArguments(
            output_dir="./llama3_math_sft",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=50,
            fp16=True,
            push_to_hub=False,
            evaluation_strategy="no",  
            report_to="none"
        )

        # SFT Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
            peft_config=self.trainer.model.peft_config if hasattr(self.trainer, 'model') else None,
            max_seq_length=512
        )

        
        print("Starting SFT training...")
        self.trainer.train()
        self.trainer.save_model("./llama3_math_sft_final")
        print("Training completed!")

    def predict(self, test_csv_path: str, output_csv_path: str):
        """Predict on the test set and generate submission files."""
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("./llama3_math_sft_final")
        self.model = AutoModelForCausalLM.from_pretrained(
            "./llama3_math_sft_final",
            device_map="auto",
            torch_dtype=torch.float16
        )

        
        _, tokenized_test, test_df = self._preprocess_data(
            train_csv_path=test_csv_path, 
            test_csv_path=test_csv_path
        )

        # pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        
        results = []
        for idx, sample in enumerate(tokenized_test):
            prompt = sample["text"]
            # （True/False）
            output = generator(
                prompt,
                max_new_tokens=5,
                temperature=0.01,
                top_p=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

            
            pred_text = output[0]["generated_text"].replace(prompt, "").strip().lower()
            is_correct = "true" in pred_text

            results.append({
                "ID": idx,
                "is_correct": is_correct
            })

            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples")

        
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_csv_path, index=False)
        print(f"Submission file saved to {output_csv_path}")


def run_math_sft_pipeline(
        train_csv: str,
        test_csv: str,
        output_csv: str
):
    """Execute the complete SFT workflow: Training → Prediction → Generate submission files"""
    sft = Llama3MathSFT()

    
    sft.train(train_csv_path=train_csv, test_csv_path=test_csv)

    
    sft.predict(test_csv_path=test_csv, output_csv_path=output_csv)


if __name__ == "__main__":
    
    TRAIN_CSV = "train_verification1.json"  
    TEST_CSV = "test_verification2.json"  
    OUTPUT_CSV = "submission.csv"  

    
    print("Starting Llama-3 8B SFT pipeline for math answer verification...")
    run_math_sft_pipeline(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        output_csv=OUTPUT_CSV
    )
    print("All tasks completed! Submission file is ready.")
