#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken
# pip install chainlit
# !chainlit hello
# !pip install python-dotenv
# !pip install huggingface hub
from langchain.llms import OpenAI
from google.colab import drive
from chainlit import cl
from langchain import HuggingFaceHub,PromptTemplate,LLMChain 
from dotenv import load_dotenv
from getpass import getpass

# drive.mount('/content/gdrive')
# load_dotenv('/content/drive/MyDrive/mykey.env') # use this if google drive mounted in colab notebook

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN


model_id='aipradeepd/Evolent_BERT_Classifier'
evolent_model = HuggingFaceHub(repo_id='aipradeepd/Evolent_BERT_Classifier',model_kwargs={'temperature':0})

template = """ You are a Medical Assistant that predicts the class of Medical test based 
on abstract received as a input. 

{abstract}
"""

@cl_on_chat_start
def main():
    prompt = PromptTemplate(template=template,input_variables=['abstract'])
    model_chain = LLMChain(llm=evolent_model,prompt=prompt,verbose=True)

    cl.user_session.set('evolent_chain',model_chain)

@cl.on_message
async def main(message:str):
    evolent_chain = cl.user_session.get('evolent_chain')
    res = await evolent_chain.acall(message,callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res['text']).send()
  

