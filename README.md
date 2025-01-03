# Final project : DS 5690-01 Transformers/ Gen AI Models class
## Project Vietnews
December 2024 -
Donna (Thuc Doan Nguyen)

## Video Recording (2 parts)

Please watch part 1 of recording here (This was created by AI by uploading transcript!): https://drive.google.com/file/d/1Jq8rnbv4FeaXqawzN3-JdQiS5vjnbFp8/view?usp=sharing

Part 2: [https://drive.google.com/file/d/1Jq8rnbv4FeaXqawzN3-JdQiS5vjnbFp8/view?usp=sharing](https://drive.google.com/file/d/1xsVGYMS60ASFN1n_HsBD7-1G7pWwW91G/view?usp=sharing)

## Overview
Context: using a pre-trained model for Vietnamese to finetune on news dataset.

 The success of pretrained seq2seq models has largely been limited to the English language. Not until around roughly 2 years ago that we had our first public large-scale  monolingual seq2seq model pre-trained for Vietnamese. 

Problem: 

Based on the seq2seq denoising autoencoder BART, BARTpho was recently introduced with two versions: BARTpho-syllable and BARTpho-word to serve as the first public large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. However, these models were not trained for newspaper articles- which contain more complex and professional words in the Vietnamese language.  In this project, I seek to finetune one of these versions with dataset focused on Vietnamese news and create an interactive tool for the task of Vietnamese text summarization and then translation to English. This tool can be used by aspiring Vietnamese learners who are attempting to master the language further by reading the latest news. Imagine clicking on a newspaper website full of unknown articles with unfamiliar jargons that, at a glance, may seem too daunting for a non-native speaker. This new tool can motivate the learners to be interested in a certain topic if they can quickly get a brief idea what the news article was writing about. The readers may at least start learning to translate from the brief summary and hopefully are enticed enough to read on to the full article.

 
Approach:

BARTpho was used as a strong baseline for research and applications of generative natural language processing (NLP) tasks for Vietnamese. It uses the "large" architecture and the pre-training scheme of the sequence-to-sequence denoising autoencoder BART, thus it is especially suitable for generative NLP tasks. I have tried some older Vietnamese NLP models but still found BARTpho performs best overall. Testing between the 2 versions of BARTpho, I chose BARTpho-syllable as it runs faster /fewer parameters (without sacrificing noticeable drop in performance).

<img width="592" alt="image" src="https://github.com/user-attachments/assets/460ea446-89ce-4111-affe-90381dbd1d1c">


I finetuned the pretrained model using my vietnews dataset (list of Vietnamese news articles from "VNDS: A Vietnamese Dataset for Summarization") and used peft library for LoRA. I used GoogleTranslator for translation from Vietnamese to English and gradio to construct an interface for users to paste the news text and then receive the summary output along with its english translation.

![image](https://github.com/user-attachments/assets/fde3d06b-5f8e-457a-87de-8bfab03eeb17)

 
How the problem was addressed:

This app serves as a helpful tool for prospective learners of the Vietnamese language, motivating them to read longer passages and latest news in topics that they may be interested in (instead of being lost in a maze of information). The users of the app also have the option to choose whether they want to see the english translation or not, so they can translate the summarized version on their own first before seeing the correct translation.

Uses: Vietnamese language news summarization and transaltion tool

Sources:


Nguyen, V. H., Nguyen, T. C., Nguyen, M. T., & Hoai, N. X. (2019). VNDS: A Vietnamese Dataset for Summarization. In 2019 6th NAFOSTED Conference on Information and Computer Science (NICS). https://doi.org/10.1109/NICS48868.2019.9023886

Phan, L., Tran, H., Nguyen, H., & Trinh, T. H. (2022). ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation. arXiv:2205.06457v2.

Tran, N. L., Le, D. M., & Nguyen, D. Q. (2021). BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese. arXiv:2109.09701v3.

Permission (VinAI)

Copyright (c) 2021 VinAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Impact

This tool shows further application of BARTpho in generative NLP tasks for Vietnamese. As a Vietnamese myself, this project was fun to work with (enhancing my own coding abilities) and it aimed to create a helpful tool for people to learn the language (like my husband/ nephew).

## Next step

It would be really cool if this can somehow incorporate audio - maybe allowing the capabilities of output as audio on top of text or allowing users to upload a recording instead.

## Links

https://huggingface.co/vinai/bartpho-syllable

https://github.com/VinAIResearch/BARTpho

https://github.com/vietai/ViT5/tree/main

https://huggingface.co/docs/peft/main/en/developer_guides/lora

https://medium.com/@shitalnandre108/fine-tuning-llama-2-large-language-model-with-custom-datasets-using-google-colab-a-comprehensive-a9d68faf3bc9
