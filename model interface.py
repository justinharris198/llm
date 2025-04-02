import os
from dotenv import load_dotenv
import ollama
from openai import OpenAI
import json
from IPython.display import Markdown, display, update_display
import anthropic
import google.generativeai
import gradio as gr

load_dotenv(override=True)
load_dotenv(override=True)
os.getenv('OPENAI_API_KEY')
os.getenv('ANTHROPIC_API_KEY')
os.getenv('GOOGLE_API_KEY')

openai = OpenAI()
anthropic.Anthropic()
google.generativeai.configure()

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
        
    def getOpenAiResponse(self, model = 'gpt-4o', responseFmt = 'text', temperature = 1.0):
        if responseFmt == 'json':
            responseLabel = {"type": "json_object"}
        else:
            responseLabel = {"type": "text"}
        
        openAiResponse = self.openai.chat.completions.create(
            model = model,
            messages = self.openAiMessage,
            temperature=temperature,
            response_format=responseLabel
        )
        
        if responseFmt == 'json':
            return json.loads(openAiResponse.choices[0].message.content)
        else:
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
    
    def getAnthropicResponse(self, model = 'claude-3-5-sonnet-latest', temperature = 1.0):
        response = self.claude.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=self.anthropicMessage['system'],
            messages=self.anthropicMessage['messages']
        )
        
        return response.content[0].text

    def getGeminiResponse(self, model = 'gemini-2.0-flash-exp'):
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
        
    def getResponseOpenAiConstruct(self, systemPrompt, userPrompt, model = 'OpenAI: gpt-4o-mini'):
        prompts = {'system': systemPrompt, 'user': userPrompt}
        aiModel = model[model.index(': ') + 2:]
        
        if 'OpenAI' in model:
            self.setOpenAiPrompts(prompts)
            
            return self.getOpenAiResponse(aiModel)
        
        if 'Anthropic' in model:
            self.setAnthropicPrompts(prompts)
            
            return self.getAnthropicResponse(aiModel)
        
        if 'Google' in model:
            self.setGeminiPrompts(prompts)
            
            return self.getGeminiResponse(aiModel)

#create an AI chat interface for various models

llm = Llm()

modelDropdown = [
    'OpenAI: gpt-4o',
    'Anthropic: claude-3-5-sonnet-latest',
    'Google: gemini-2.0-flash-exp'
]
defaultSystemPrompt = '''
You are a helpful, knowledgeable AI assistant. Please follow these guidelines:
1. Provide accurate, clear, and concise information.
2. If a query is ambiguous or missing details, ask clarifying questions.
3. Remain neutral and factual; don’t include personal opinions or biases.
4. If you don’t know an answer, say so rather than providing incorrect or speculative responses.
5. Be polite and professional.
'''

inputs = [
    gr.Textbox(label="System Prompt:", lines=6, value = defaultSystemPrompt),
    gr.Textbox(label="User Prompt:", lines=6),
    gr.Dropdown(choices=modelDropdown, label="Model")
]

view = gr.Interface(
    fn=llm.getResponseOpenAiConstruct,
    inputs=inputs,
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never"
)
view.launch()