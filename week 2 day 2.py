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
        
    def getOpenAiResponse(self, model = 'gpt-4o', responseFmt = None, temperature = 1.0):
        if responseFmt == 'json':
            openAiResponse = self.openai.chat.completions.create(
                model = model,
                messages = self.openAiMessage,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        else:
            openAiResponse = self.openai.chat.completions.create(
                model = model,
                messages = self.openAiMessage,
                temperature=temperature
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
    
    def getAnthropicResponse(self, model = 'claude-3-5-sonnet-latest', responseFmt = 'text', temperature = 1.0):
        response = self.claude.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=self.anthropicMessage['system'],
            messages=self.anthropicMessage['messages']
        )
        
        if responseFmt == 'json':
            try:
                return json.loads(response.content[0].text)
            except:
                print(response.content[0].text)
                return response.content[0].text
        
        return response.content[0].text

    def getGeminiResponse(self, model = 'gemini-2.0-flash-exp', responseFmt = 'text'):
        geminiResponse = OpenAI(
            api_key=self.geminiKey, 
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        if responseFmt == 'json':
            geminiResponse = geminiResponse.chat.completions.create(
                model=model,
                messages=self.geminiMessage,
                response_format = {"type": "json_object"}
            )
        else:
            geminiResponse = geminiResponse.chat.completions.create(
                model=model,
                messages=self.geminiMessage
        )
            
        if responseFmt == 'json':
            return json.loads(geminiResponse.choices[0].message.content)
        else:
            return geminiResponse.choices[0].message.content

        
    def getResponseOllama(self, model):
        self.ollamaResponse = ollama.chat(model=model, messages=self.message)
        
    def getResponseOpenAiConstruct(self, systemPrompt, userPrompt, model = 'OpenAI: gpt-4o-mini', responseFmt = 'text'):
        prompts = {'system': systemPrompt, 'user': userPrompt}
        aiModel = model[model.index(': ') + 2:]
        
        if 'OpenAI' in model:
            self.setOpenAiPrompts(prompts)
            
            return self.getOpenAiResponse(aiModel, responseFmt = responseFmt)
        
        if 'Anthropic' in model:
            self.setAnthropicPrompts(prompts)
            
            return self.getAnthropicResponse(aiModel, responseFmt = responseFmt)
        
        if 'Google' in model:
            self.setGeminiPrompts(prompts)
            
            return self.getGeminiResponse(aiModel, responseFmt = responseFmt)

#create an AI chat interface for various models
'''
llm = Llm()

modelDropdown = [
    'OpenAI: gpt-4o',
    'Anthropic: claude-3-5-sonnet-latest',
    'Google: gemini-2.0-flash-exp'
]

inputs = [
    gr.Textbox(label="System Prompt:", lines=6),
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
'''

class ScrapeWebData:
    def __init__(self, url):
        self.url = url
        
        driver = webdriver.Chrome()  # or Firefox, Edge, etc.
        driver.get(url)
        
        html = driver.page_source
        self.body = html
        soup = BeautifulSoup(self.body, 'html.parser')
        driver.quit()
        
        self.title = soup.title.string if soup.title else "No title found"
        
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        
    def getContents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def buildLinkPrompts(data):
    systemPrompt = "You are provided with a list of links found on a webpage. \
    You are able to decide which of the links would be most relevant to include in a brochure about the company, \
    such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
    systemPrompt += "You should respond in JSON as in this example:"
    systemPrompt += """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page": "url": "https://another.full.url/careers"}
        ]
    }
    """

    userPrompt = f"Here is the list of links on the website of {data.url} - "
    userPrompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
    userPrompt += "Links (some might be relative links):\n"
    userPrompt += "\n".join(data.links)
    
    return {'system': systemPrompt, 'user': userPrompt}

def buildProductPrompts(details, name):
    systemPrompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
    and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
    Include details of company culture, customers and careers/jobs if you have the information. The response should be in valid Markdown"


    userPrompt = f"You are looking at a company called: {name}\n"
    userPrompt += "Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    userPrompt += details
    userPrompt = userPrompt[:5000] # Truncate if more than 5,000 characters
    
    return {'system': systemPrompt, 'user': userPrompt}

def buildUserPromptWebsiteData(data, links):
    result = "Landing page:\n"
    result += data.getContents()

    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += ScrapeWebData(link["url"]).getContents()
    
    return result    

def buildCatalog(website, title, model):
    llm = Llm()
    data = ScrapeWebData(website)

    prompts = buildLinkPrompts(data)
    
    links = llm.getResponseOpenAiConstruct(
        prompts['system'], 
        prompts['user'], 
        model = model, 
        responseFmt = 'json'
    )
    userPromptData = buildUserPromptWebsiteData(data, links)
    
    prompts = buildProductPrompts(userPromptData, title)
    
    return llm.getResponseOpenAiConstruct(prompts['system'], prompts['user'], model)

modelDropdown = [
    'OpenAI: gpt-4o',
    'Anthropic: claude-3-5-sonnet-latest',
    'Google: gemini-2.0-flash-exp'
]

inputs = [
    gr.Textbox(label="Website:", lines=1),
    gr.Textbox(label="Title", lines=1),
    gr.Dropdown(choices=modelDropdown, label="Model")
]

view = gr.Interface(
    fn=buildCatalog,
    inputs=inputs,
    outputs=[gr.Textbox(label="Response:")],
    flagging_mode="never"
)
view.launch()
