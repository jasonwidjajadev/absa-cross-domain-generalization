# CONTRIBUTION.md

**(a) Team Name:** Cold Tuna

---

**(b) Team Members:**
- Jason Widjaja, z5494973, z5494973@ad.unsw.edu.au
- Mingyang Cai, z5402186, z5402186@ad.unsw.edu.au
- Hamzah Alamdeen, z5428209, z5428209@ad.unsw.edu.au
- Cheng Zeng, z5431483, z5431483@ad.unsw.edu.au
- Bryan Che-Sheng Jen, z5360176, z5360176@ad.unsw.edu.au

---

**(c) A separate description of work done by every member:**

### Jason Widjaja (z5494973)
- Dataset split preparation for semeval 2014 restaurant and laptop dataset, graph for dataset distributions
- Conducted supervised fine tuning on Bert-base-uncased and evaluation on restaurant and laptop dataset, gradio demo code
- Report: worked on Bert methodologies, result writeup.
- Worked graphs of 4 models comparisons and conclusions of report
- Conducted deep linguistic error analysis (sentence level missclassifications) of the 4 comparable models for discussions of report on both restaurant and laptop dataset
- Worked on the presentation slides (old slides reused in new slides), methodologies, bert, comparison results and conclusions

### Mingyang Cai (z5402186)
- Custom Dataset Preparation: Went through 750 reviews from Yelp and annotated multiple aspects and its sentiments from each review. Then went through another 750 reviews annotated by Hamzah and provided my own sentiment annotation for them. Went through dataset evaluation
- Qwen SFT (Laptop Dataset): Conducted supervised fine-tuning (SFT) of the Qwen model on the laptop dataset and performed evaluation on the fine-tuned model.
- Report: Worked on the abstract and related work section as well as all other sections related to Qwen method in mehtodology and results.
- Presentation slides: Contributed to formatting the presentation slides and dataset-related sections.

### Hamzah Alamdeen (z5428209)
- Custom Dataset Preparation: Went through 750 reviews from Yelp and annotated multiple aspects and its sentiments from each review. Then went through another 750 reviews annotated by Mingyang and provided my own sentiment annotation for them.
- Chain Of Thought Prompting method: Developed the core code for the method as well as the evaluation on both restaurant dataset and laptop dataset.
- Bert Fine-tuned on laptop dataset: Fine tuned the Bert Uncased that was trained on the restaurant dataset to train on the laptop dataset, did evaluation on the new fine tuned model.
- Gradio Demo: Helped to write the stubs and base for the gradio demo file.
- Report: Worked on the introduction section as well as all other sections related to CoT Prompting Method in mehtodology and results.
- Presentation slides: Contributed to formatting the presentation slides and CoT section.

### Cheng Zeng (z5431483)
- Custom Dataset Preparation: Reviewed and annotated 750 Yelp reviews for aspect-level sentiment labels, then cross-checked another 750 reviews annotated by a teammate and provided an additional sentiment annotation pass; also organized and cleaned the custom dataset and exported sentiment polarity disagreement cases for further analysis and annotation quality checking.
- Qwen SFT (Restaurant Dataset): Conducted supervised fine-tuning (SFT) of the Qwen model on the restaurant dataset and performed evaluation on the fine-tuned model.
- Report: Wrote the report sections related to the Qwen method, including methodology and results.
- Presentation slides: Contributed to the Qwen and dataset-related sections of the presentation slides.

### Bryan Che-Sheng Jen (z5360176)
- Custom Dataset Preparation: Reviewed and annotated 750 Yelp reviews for aspect-level sentiment labels, then cross-checked another 750 reviews annotated by a teammate and provided an additional sentiment annotation pass; also organized and cleaned the custom dataset and exported sentiment polarity disagreement cases for further analysis and annotation quality checking.
- TF-IDF + Logistic Regression: Contributed to the TF-IDF + Logistic Regression baseline for aspect-level sentiment classification, including implementation and analysis.
- Report: Contributed to the dataset section, the TF-IDF + Logistic Regression section, and made contributions throughout the report, including editing, restructuring, and improving clarity across multiple sections.
- Presentation slides: Contributed to the TF-IDF + LR sections of the presentation slides.
