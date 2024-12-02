from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from deep_translator import GoogleTranslator
import re
import os
import logging
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#me
training_args = Seq2SeqTrainingArguments(
        output_dir="./output",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=15,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        )
def load_vietnews_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt.seg"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    summary = lines[0].strip()
                    content = ' '.join(line.strip() for line in lines[1:])
                    data.append({"summary": summary, "content": content})
    return data

# me


class VietnameseSummarizer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_name = "vinai/bartpho-syllable"
        logger.info(f"Loading model (fine-tuned)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self.translator = GoogleTranslator(source='vi', target='en')
        
        self.model.eval()
        logger.info("Model loaded successfully")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\đĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ,.-]', ' ', text)
        return ' '.join(text.split())

    # me
    peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]  # Targeting attention projections
        )
    #tokenized_dataset = dataset.map(preprocess_function, batched=True)


    #trainer = SFTTrainer(
    #model=model,
    #train_dataset=vietnews_data,
    #peft_config=peft_config,
    #dataset_text_field="text",
    #max_seq_length=max_seq_length,
    #tokenizer=tokenizer,
    #args=training_arguments,
    #packing=packing,
#)


    #trainer.train()
    logger.info("Fine-tuning completed!")
    #self.model.save_pretrained(output_dir)
    #self.tokenizer.save_pretrained(output_dir)
    # me

    def generate_summary(self, text: str, translate_to_english: bool = True) -> dict:
        try:
            if not text.strip():
                return {
                    'vietnamese_summary': "Vui lòng nhập văn bản để tóm tắt.",
                    'english_summary': "Please enter text to summarize."
                }

            cleaned_text = self.clean_text(text)
            input_text = f"tóm tắt tin tức: {cleaned_text}"

            inputs = self.tokenizer(
                input_text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    min_length=30,
                    num_beams=5,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    top_p=0.95,
                    temperature=0.7,
                    do_sample=True #False
                )

            vietnamese_summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            vietnamese_summary = self.clean_text(vietnamese_summary)
            
            # Remove "Tóm tắt" at the start if it exists
            if vietnamese_summary.lower().startswith("tóm tắt"):
                vietnamese_summary = vietnamese_summary[len("tóm tắt"):].strip()

            # Adjust prefix "tin tức" to "tin tức: "
            if vietnamese_summary.lower().startswith("tin tức"):
                vietnamese_summary = "tin tức: " + vietnamese_summary[len("tin tức"):].strip()

            # Ensure the last sentence is excluded but preserve periods
            vietnamese_sentences = vietnamese_summary.split('.')
            if len(vietnamese_sentences) > 1:
                vietnamese_summary = '. '.join(vietnamese_sentences[:-1]).strip() + '.'

            english_summary = ""
            if translate_to_english:
                try:
                    english_summary = self.translator.translate(vietnamese_summary)
                    
                    # Remove "Summary" at the start if it exists
                    if english_summary.lower().startswith("summary"):
                        english_summary = english_summary[len("summary"):].strip()

                    # Adjust prefix "news" to "news: "
                    if english_summary.lower().startswith("news"):
                        english_summary = "news: " + english_summary[len("news"):].strip()

                    # Ensure the last sentence is excluded but preserve periods
                    english_sentences = english_summary.split('.')
                    if len(english_sentences) > 1:
                        english_summary = '. '.join(english_sentences[:-1]).strip() + '.'

                except Exception as e:
                    logger.error(f"Translation error: {str(e)}")
                    english_summary = "Translation failed"

            return {
                'vietnamese_summary': vietnamese_summary,
                'english_summary': english_summary
            }

        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return {
                'vietnamese_summary': f"Lỗi tóm tắt: {str(e)}",
                'english_summary': f"Summarization error: {str(e)}"
            }

    def create_interface(self):
        with gr.Blocks(title="Vietnamese Text Summarizer") as interface:
            gr.Markdown("# Vietnamese Text Summarizer / Công cụ Tóm tắt Văn bản Tiếng Việt")
            
            with gr.Row():
                input_text = gr.Textbox(
                    label="Input Text / Văn bản đầu vào",
                    placeholder="Enter Vietnamese text here / Nhập văn bản tiếng Việt vào đây",
                    lines=10
                )
            
            with gr.Row():
                translate_checkbox = gr.Checkbox(
                    label="Generate English translation / Tạo bản dịch tiếng Anh",
                    value=True
                )
            
            with gr.Row():
                summarize_btn = gr.Button("Generate Summary / Tạo bản tóm tắt")
            
            with gr.Row():
                vietnamese_output = gr.Textbox(label="Vietnamese Summary / Bản tóm tắt tiếng Việt", lines=5)
                english_output = gr.Textbox(label="English Summary / Bản tóm tắt tiếng Anh", lines=5)

            def summarize_wrapper(text, translate):
                results = self.generate_summary(text, translate)
                return results['vietnamese_summary'], results['english_summary']

            summarize_btn.click(
                fn=summarize_wrapper,
                inputs=[input_text, translate_checkbox],
                outputs=[vietnamese_output, english_output]
            )

        return interface

def main():
    try:
        logger.info("Starting Vietnamese Summarizer...")
        summarizer = VietnameseSummarizer()
        interface = summarizer.create_interface()
        interface.launch(share=True)
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

if __name__ == "__main__":
    main()
