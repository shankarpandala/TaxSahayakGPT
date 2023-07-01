import os
import openai
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

openai.api_key = os.environ['OPENAI_API_KEY']
UPLOAD_FOLDER = 'uploads'


llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryMemory(llm=OpenAI()),
    verbose=True
)


# python function to generate openai completion responses

def generate_response(user_input):
    # set the temperature low to get more coherent results
    response = conversation_with_summary.predict(input=str(user_input))
    print("RESPONSE from chatbot.py:", response)
    return response

if __name__ == '__main__':
    generate_response("Hello, my name is John. What is your name?")