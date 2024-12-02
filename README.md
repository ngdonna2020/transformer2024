# Final project : DS 5690-01 Transformers/ Gen AI Models class
## Project Vietnews
December 2024 -
Donna (Thuc Doan Nguyen)

## Overview
Context: using a pre-trained model for Vietnamese to finetune on news dataset.

 The success of pretrained seq2seq models has largely been limited to the English language. Not until around roughly 2 years ago that we had our first public large-scale  monolingual seq2seq model pre-trained for Vietnamese. 

Problem: 

Based on the seq2seq denoising autoencoder BART, BARTpho was introduced with two versions: BARTpho-syllable and BARTpho-word to serve as the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. BARTpho uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, thus it is especially suitable for generative NLP tasks.  In this project, I seek to finetune one of these versions with dataset focused on Vietnamese news and create an interactive tool for the task of Vietnamese text summarization and then translation to English. This tool can be used by aspiring Vietnamese learners who are attempting to master the language by reading the latest news (like my husband). Imagine clicking on a newspaper website full of unknown articles with unfamiliar jargons that, at a glance, may be too daunting for a non-native speaker. This new tool can motivate them to be interested in a certain topic if the reader can quickly get a brief idea what the news article is written about. They may at least start translating from the brief summary and hopefully enticed enough to read on to the full article.

 
Approach
BARTpho was used as a strong baseline for research and applications of generative natural language processing (NLP) tasks for Vietnamese. gradio
 
How the problem was addressed:

Testing between the 2 versions of BARTpho, I chose BARTpho-syllable as it runs faster /fewer parameters (without sacrificing noticeable drop in performance)

uses, sources, permissions, code

## Impact

## Links
links of where to go to get more information (other papers, models, blog posts (e.g. papers with code)
