from openai import OpenAI

# TODO turn into common State class which passes to all instances?
testing_mode = True

class Summarizer:
    def __init__(self):
        self.API_KEY = "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"
        self.client = OpenAI(api_key=self.API_KEY)

    def summarize(self, text):
        if testing_mode:
            summary = "The research papers explore machine learning algorithms, categorizing and comparing their performance in supervised settings. They survey applications of these algorithms, focusing on the fundamental components and principles of machine learning workflows that handle both numerical and categorical data. This summary highlights essential insights into algorithm effectiveness and data processing methodologies."
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"You are a researcher, accurately summarize the following set of research papers into a single paragraph: {text}"}
                ]
            )
            summary = response.choices[0].message.content

        return summary

