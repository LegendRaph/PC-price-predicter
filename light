import os
import fitz
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# GUI



print("** Summarizer **")

path = input("Enter the file path: ")
file_path = os.path.join("C:\\Users\\adele\\Documents", path)

if path.endswith('.txt'):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=int(len(parser.document.sentences) * 0.5))
            for sentence in summary:
                print("\nSummary: ")
                print(sentence)

    except FileNotFoundError:
        print("Error: File not found.")

elif path.endswith('.pdf'):
    try:
        with fitz.open(file_path) as pdf:
            ptext = "\n".join(page.get_text() for page in pdf)
            parser = PlaintextParser.from_string(ptext, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=int(len(parser.document.sentences) * 0.5))
            for sentence in summary:
                print("\nSummary: ")
                print(sentence)

    except FileNotFoundError:
        print("Error: File not found.")




# Summarizer

