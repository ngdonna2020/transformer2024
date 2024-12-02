# Final project : DS 5690-01 Transformers/ Gen AI Models class
## Project Vietnews
December 2024 -
Donna (Thuc Doan Nguyen)

## Overview
Context: using a pre-trained model for Vietnamese to finetune on news dataset.

 The success of pretrained seq2seq models has largely been limited to the English language. Not until around roughly 2 years ago that we had our first public large-scale  monolingual seq2seq model pre-trained for Vietnamese. As a Vietnamese myself, this project was fun to work with and hopefully created a helpful tool for people to learn the language.

Problem: 

Based on the seq2seq denoising autoencoder BART, BARTpho was recently introduced with two versions: BARTpho-syllable and BARTpho-word to serve as the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese.  In this project, I seek to finetune one of these versions with dataset focused on Vietnamese news and create an interactive tool for the task of Vietnamese text summarization and then translation to English. This tool can be used by aspiring Vietnamese learners who are attempting to master the language further by reading the latest news (like my husband). Imagine clicking on a newspaper website full of unknown articles with unfamiliar jargons that, at a glance, may seem too daunting for a non-native speaker. This new tool can motivate the learners to be interested in a certain topic if they can quickly get a brief idea what the news article was writing about. The readers may at least start learning to translate from the brief summary and hopefully are enticed enough to read on to the full article.

 
Approach:

BARTpho was used as a strong baseline for research and applications of generative natural language processing (NLP) tasks for Vietnamese. It uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, thus it is especially suitable for generative NLP tasks. I have tried some older Vietnamese text summarizer models but still found BARTpho performs best overall. Testing between the 2 versions of BARTpho, I chose BARTpho-syllable as it runs faster /fewer parameters (without sacrificing noticeable drop in performance). I finetuned the pretrained model using my vietnews dataset (list of Vietnamese news articles stored in .txt.seg files) and used peft library for LoRA configuration (which speeds up finetuning and uses less memory). I used GoogleTranslator for translation from Vietnamese to English and gradio to construct a simple interface for users to paste the news text and then receive the summary output along with its english translation.
 
How the problem was addressed:

This app shows further application of BARTpho in generative NLP tasks for Vietnamese. It serves as a helpful tool for prospective learners of the Vietnamese language, motivating them to read longer passages and latest news in topics that they may be interested in (instead of being lost in a maze of information). The users of the app also have the option to choose whether they want to see the english translation or not, so they can translate the summarized version on their own first before seeing the correct translation.

uses, sources, permissions

## Impact

## Links
links of where to go to get more information (other papers, models, blog posts (e.g. papers with code)
