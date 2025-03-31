import os
from dotenv import load_dotenv
from selenium import webdriver
from bs4 import BeautifulSoup
import ollama
from openai import OpenAI
import json
from IPython.display import Markdown, display, update_display
import anthropic
import google.generativeai

load_dotenv(override=True)

class Llm:
    def __init__(self):
        self.openaiKey = os.getenv('OPENAI_API_KEY')
        self.anthropicKey = os.getenv('ANTHROPIC_API_KEY')
        self.geminiKey = os.getenv('GOOGLE_API_KEY')
        
        self.openai = OpenAI()
        self.claude = anthropic.Anthropic()
        google.generativeai.configure()

    def setAnthropicPrompts(self, prompts, append = False):
        if not append:
            self.anthropicMessage = {'system': prompts['system'], 'messages': []}
        
        for i in prompts:
            if i != 'system':
                self.anthropicMessage['messages'].append({"role": i, "content": prompts[i]})
                
    def setOpenAiPrompts(self, prompts, append = False):
        if not append:
            self.openAiMessage = [{"role": "system", "content": prompts['system']}]
        
        for i in prompts:
            if i != 'system':
                self.openAiMessage.append({"role": i, "content": prompts[i]})
                
    def setGeminiPrompts(self, prompts, append = False):
        if not append:
            self.geminiMessage = [{"role": "system", "content": prompts['system']}]
        
        for i in prompts:
            if i != 'system':
                self.geminiMessage.append({"role": i, "content": prompts[i]})
        
    def getOpenAiResponse(self, model, temperature = 1.0):
        openAiResponse = self.openai.chat.completions.create(
            model = model,
            messages = self.openAiMessage,
            temperature=temperature
        )
        
        return openAiResponse.choices[0].message.content
        
    def getOpenAiStreamedResponse(self, model = 'gpt-4o'):
        stream = self.openai.chat.completions.create(
            model = model,
            messages = self.openAiMessage,
            stream = True
        )

        response = ''
        display_handle = display(Markdown(''), display_id=True)
        
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            response = response.replace("```","").replace("markdown", "")
            update_display(Markdown(response), display_id=display_handle.display_id)

    def getResponseOpenAIJson(self, model):
        openAiResponse = self.openai.chat.completions.create(
            model = model,
            messages = self.message,
            response_format={"type": "json_object"}
        )
        
        openAiResponseJson = openAiResponse.choices[0].message.content
        self.openAiResponseJson = json.loads(openAiResponseJson)
    
    def getAnthropicResponse(self, model, temperature = 1.0):
        response = self.claude.messages.create(
            model=model,
            max_tokens=200,
            temperature=temperature,
            system=self.anthropicMessage['system'],
            messages=self.anthropicMessage['messages']
        )
        
        return response.content[0].text

    def getGeminiResponse(self, model):
        geminiResponse = OpenAI(
            api_key=self.geminiKey, 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        geminiResponse = geminiResponse.chat.completions.create(
            model=model,
            messages=self.geminiMessage
        )
        
        return geminiResponse.choices[0].message.content

        
    def getResponseOllama(self, model):
        self.ollamaResponse = ollama.chat(model=model, messages=self.message)
        
llm = Llm()
prompts = {
    'system': "You are an assistant that is great at telling jokes",
    'user': "Tell a light-hearted joke for an audience of Data Scientists"
}
llm.setAnthropicPrompts(prompts)
llm.setOpenAiPrompts(prompts)
llm.setGeminiPrompts(prompts)

print(llm.getOpenAiResponse('gpt-3.5-turbo', 0.7))
print(llm.getOpenAiResponse('gpt-4o-mini', 0.7))
print(llm.getOpenAiResponse('gpt-4o', 0.7))
print(llm.getAnthropicResponse('claude-3-5-sonnet-latest', 0.7))
print(llm.getGeminiResponse('gemini-2.0-flash-exp'))
'''
prompts = {
    'system': "You are a helpful assistant that responds in Markdown",
    'user': "How do I decide if a business problem is suitable for an LLM solution? Please respond in Markdown."
}
llm.setOpenAiPrompts(prompts)
llm.getOpenAiStreamedResponse()
'''
gpt_model = "gpt-4o-mini"
claude_model = "claude-3-haiku-20240307"

gpt_system = "you are very opinionated in this convervsation."
claude_system = "you are very shy, but don't agree with what the other person is saying to you"

gptPrompt = {
    'system': "You are a chatbot who is very argumentative; \
        you disagree with anything in the conversation and you challenge everything, in a snarky way.You are a helpful assistant that responds in Markdown",
    'assistant': "hi",
    'user': 'hi there'
}
    
anthropicPrompt = {
    'system': "You are a very polite, courteous chatbot. You try to agree with \
    everything the other person says, or find common ground. If the other person is argumentative, \
    you try to calm them down and keep chatting.",
    'user': 'hi there', 
    'assistant': "hi"
}

llm.setOpenAiPrompts(gptPrompt)
llm.setAnthropicPrompts(anthropicPrompt)

for i in range(5):
    a = llm.getOpenAiResponse("gpt-4o-mini")
    llm.setAnthropicPrompts({'user': a}, append = True)
    llm.setOpenAiPrompts({'assistant': a}, append = True)
    print(a)
    b = llm.getAnthropicResponse("claude-3-haiku-20240307")
    llm.setAnthropicPrompts({'assistant': b}, append = True)
    llm.setOpenAiPrompts({'user': b}, append = True)
    print(b)
    llm.anthropicMessage
    
gpt_model = "gpt-4o-mini"
claude_model = "claude-3-haiku-20240307"
gemini_model = "gemini-2.0-flash-exp"

gpt_system = "You are speaking to 2 people about the future of AI. You must explain why AI is useful. You are speaking to Gemini and Claude, 2 tech experts."
claude_system = "You are speaking to 2 people about the future of AI. You must help the other 2 see why AI could be harmful. You are speaking to openAi and gemini, 2 tech experts."
gemini_system = "You are speaking to 2 people about the future of AI. It's hard for you to take this topic seriously. You are speaking to openAi and claude, 2 tech experts."

gptPrompt = {
    'system': gpt_system
}
  
anthropicPrompt = {
    'system': claude_system
}
geminiPrompt = {
    'system': gemini_system
}

llm.setOpenAiPrompts(gptPrompt)
llm.setGeminiPrompts(geminiPrompt)
llm.setAnthropicPrompts(anthropicPrompt)

gpt = 'hi'
claude = 'hello'
gemini = 'hi there'

llm.setOpenAiPrompts({'assistant': gpt}, append = True)
llm.setOpenAiPrompts({'user': 'claude says: ' + claude}, append = True)
llm.setOpenAiPrompts({'user': 'gemini says : ' + gemini}, append = True)

llm.setAnthropicPrompts({'user': 'opan ai says: ' + gpt}, append = True)
llm.setAnthropicPrompts({'assistant': claude}, append = True)
llm.setAnthropicPrompts({'user': 'gemini says: ' + gemini}, append = True)

llm.setGeminiPrompts({'user': 'opan ai says: ' + gpt}, append = True)
llm.setGeminiPrompts({'user': 'claude says: ' + claude}, append = True)
llm.setGeminiPrompts({'assistant': gemini}, append = True)

for i in range(5):
    gpt = llm.getOpenAiResponse("gpt-4o-mini")
    
    llm.setOpenAiPrompts({'assistant': gpt}, append = True)
    llm.setAnthropicPrompts({'user': 'opan ai says: ' + gpt}, append = True)
    llm.setGeminiPrompts({'user': 'opan ai says: ' + gpt}, append = True)
    
    print(gpt)
    print('')
    
    claude = llm.getAnthropicResponse("claude-3-haiku-20240307")
    llm.setOpenAiPrompts({'user': 'claude says: ' + claude}, append = True)
    llm.setAnthropicPrompts({'assistant': claude}, append = True)
    llm.setGeminiPrompts({'user': 'claude says: ' + claude}, append = True)
    
    print(claude)
    print('')
    gemini = llm.getGeminiResponse('gemini-2.0-flash-exp')
    llm.setOpenAiPrompts({'user': 'gemini says : ' + gemini}, append = True)
    llm.setAnthropicPrompts({'user': 'gemini says: ' + gemini}, append = True)
    llm.setGeminiPrompts({'assistant': gemini}, append = True)
    
    print(gemini)
    print('')
    
    
    
    
