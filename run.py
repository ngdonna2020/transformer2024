import os
import logging
from vietnamese_summarizer import VietnameseSummarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Initializing Vietnamese Summarizer...")
        device = "cpu"
        logger.info(f"Using device: {device}")
        
        summarizer = VietnameseSummarizer(device=device)
        logger.info("Model loaded successfully")
        
        logger.info("Starting Gradio interface...")
        interface = summarizer.create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

if __name__ == "__main__":
    main()