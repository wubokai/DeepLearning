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
        """加载Llama-3Tokenizer并配置"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # 避免生成时警告
        self.tokenizer.add_eos_token = True
        return self.tokenizer

    def _load_model(self):
        """加载量化后的Llama-3 8B模型"""
        # 4-bit量化配置（降低显存占用）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # 自动分配GPU/CPU
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # LoRA配置（轻量化微调）
        lora_config = LoraConfig(
            r=8,  # LoRA秩
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Llama-3关键模块
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  # 显示可训练参数比例
        return self.model

    def _preprocess_data(self, train_csv_path: str, test_csv_path: str):
        """预处理训练集和测试集"""
        # 加载数据
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # 数据清洗函数
        def clean_text(text: str) -> str:
            if pd.isna(text):
                return ""
            # 移除代码块
            text = re.sub(r'<llm-code>.*?</llm-code>', '', text, flags=re.DOTALL)
            text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', text, flags=re.DOTALL)
            # 标准化数字格式（去除多余空格、统一小数点）
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\.', '.', text)
            return text.strip()

        # 构建训练集prompt（指令微调格式）
        def format_train_sample(row):
            question = clean_text(row["question"])
            answer = clean_text(row["answer"])
            solution = clean_text(row["solution"])

            prompt = f"""Decide if the given answer to the math question is correct. Respond with True or False only.
Math Question: {question}
Given Answer: {answer}
Solution Explanation: {solution}
Is the answer correct? """

            # 标签转换为文本（True/False）
            label = "True" if row["is_correct"] else "False"
            return {"text": prompt + label}

        # 构建测试集prompt（无标签）
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

        # 应用格式化
        train_dataset = Dataset.from_pandas(train_df).map(format_train_sample)
        test_dataset = Dataset.from_pandas(test_df).enumerate().map(
            lambda x: format_test_sample(x["element"], x["index"])
        )

        # Tokenize数据
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        # 设置标签（与输入错位，因果LM训练）
        tokenized_train = tokenized_train.map(
            lambda x: {"labels": x["input_ids"][1:] + [self.tokenizer.pad_token_id]}
        )

        return tokenized_train, tokenized_test, test_df

    def train(self, train_csv_path: str, test_csv_path: str):
        """训练SFT模型"""
        # 加载组件
        self._load_tokenizer()
        self._load_model()

        # 预处理数据
        tokenized_train, tokenized_test, _ = self._preprocess_data(train_csv_path, test_csv_path)

        # 训练参数配置
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
            evaluation_strategy="no",  # 若需验证可改为"epoch"
            report_to="none"
        )

        # 初始化SFT Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            tokenizer=self.tokenizer,
            peft_config=self.trainer.model.peft_config if hasattr(self.trainer, 'model') else None,
            max_seq_length=512
        )

        # 开始训练
        print("Starting SFT training...")
        self.trainer.train()
        self.trainer.save_model("./llama3_math_sft_final")
        print("Training completed!")

    def predict(self, test_csv_path: str, output_csv_path: str):
        """对测试集预测并生成提交文件"""
        # 加载训练好的模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("./llama3_math_sft_final")
        self.model = AutoModelForCausalLM.from_pretrained(
            "./llama3_math_sft_final",
            device_map="auto",
            torch_dtype=torch.float16
        )

        # 预处理测试集
        _, tokenized_test, test_df = self._preprocess_data(
            train_csv_path=test_csv_path,  # 训练集路径仅为适配函数，实际不使用
            test_csv_path=test_csv_path
        )

        # 构建生成pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        # 预测每个样本
        results = []
        for idx, sample in enumerate(tokenized_test):
            prompt = sample["text"]
            # 生成预测（仅输出True/False）
            output = generator(
                prompt,
                max_new_tokens=5,
                temperature=0.01,
                top_p=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 解析输出
            pred_text = output[0]["generated_text"].replace(prompt, "").strip().lower()
            is_correct = "true" in pred_text

            results.append({
                "ID": idx,
                "is_correct": is_correct
            })

            # 打印进度
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_df)} samples")

        # 保存提交文件
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_csv_path, index=False)
        print(f"Submission file saved to {output_csv_path}")


def run_math_sft_pipeline(
        train_csv: str,
        test_csv: str,
        output_csv: str
):
    """运行完整SFT流程：训练→预测→生成提交文件"""
    sft = Llama3MathSFT()

    # 训练模型（首次运行需执行）
    sft.train(train_csv_path=train_csv, test_csv_path=test_csv)

    # 预测并生成提交文件
    sft.predict(test_csv_path=test_csv, output_csv_path=output_csv)


if __name__ == "__main__":
    # 配置文件路径（请替换为你的数据集路径）
    TRAIN_CSV = "train_verification1.json"  # 作业提供的训练集路径
    TEST_CSV = "test_verification2.json"  # 作业提供的测试集路径
    OUTPUT_CSV = "submission.csv"  # 提交文件路径

    # 运行完整流程
    print("Starting Llama-3 8B SFT pipeline for math answer verification...")
    run_math_sft_pipeline(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        output_csv=OUTPUT_CSV
    )
    print("All tasks completed! Submission file is ready.")
