import pandas as pd
import json
import os


def convert_parquet_to_json(parquet_path, json_path, is_training=True):

    try:
        df = pd.read_parquet(parquet_path)

        json_data = []
        for index, row in df.iterrows():

            instruction = f"You are a mathematical answer verification system. Please determine whether the answer to the following mathematical question is correct: \n\nQuestion: {row.get('question', '')}\nAnswer: {row.get('answer', '')}"

            if 'solution' in row and pd.notna(row['solution']):
                instruction += f"\nSolution: {row.get('solution', '')}"


            if is_training:

                if row.get('is_correct', False):
                    output = "This answer is correct. By verifying the mathematical reasoning and calculation process in the solution, it is confirmed that the answer is accurate and correct."
                else:
                    output = "This answer is incorrect. There is a mathematical reasoning or calculation error in the solution, which leads to an incorrect final answer."
            else:

                output = ""

            item = {
                "instruction": instruction,
                "output": output
            }
            json_data.append(item)


        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"Successfully converted: {parquet_path} -> {json_path}")
        print(f"Converted {len(json_data)} ")
        return True

    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def main():

    train_parquet = "./train.parquet"
    test_parquet = "./test.parquet"

    train_json = "./train_formatted.json"
    test_json = "./test_formatted.json"


    if os.path.exists(train_parquet):
        convert_parquet_to_json(train_parquet, train_json, is_training=True)


    if os.path.exists(test_parquet):
        convert_parquet_to_json(test_parquet, test_json, is_training=False)


if __name__ == "__main__":
    main()