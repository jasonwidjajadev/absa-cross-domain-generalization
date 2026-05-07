## cot.py

This is the main file for the Chain of Thought prompting method using GPT-5.4-mini.
In order to run this file you will require an OpenAI API key stored in a `.env` file in the same folder as `cot.py`:
```
OPENAI_API_KEY="enter your key here"
```
To run this file simply enter the command:
```
python cot.py
```
This will produce predicted sentiments on the restaurants and laptops test sets using both zero-shot and few-shot CoT prompting and save the outputs to csv files. Evaluation will also be done and the results will be printed to the terminal.